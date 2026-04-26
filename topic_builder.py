import pandas as pd
import numpy as np

from collections import Counter, deque, defaultdict
from typing import Sequence, Optional, Dict

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan

from nltk.corpus import wordnet as wn



class TopicBuilder:


    def __init__(self,
                synset_embeddings: pd.DataFrame,
                languages: Optional[Sequence[str] | str] = None):
        """
        Initialize the topic builder from NLTK WordNet/OMW plus external embeddings.

        Args:
            synset_embeddings: synset embeddings with one row per synset
            languages: OMW language code(s) to load for lexical forms.
                Examples: "eng", ["eng", "cmn", "spa"].
                If None, defaults to ("eng", "cmn").
        """
        self.synset_embeddings = synset_embeddings.copy()
        self.synset_words_df = self.build_synset_words_from_nltk(languages=self._normalize_languages(languages))
        self.semlinks = self.build_semlinks_from_nltk()
        self.lexicon_long = self._build_lexicon_long(self.synset_words_df)
        self.available_languages = sorted(self.lexicon_long["lang"].dropna().astype(str).unique().tolist())
        self._build_graph()

    @staticmethod
    def _normalize_languages(languages: Optional[Sequence[str] | str]) -> Sequence[str]:
        if languages is None:
            return ("eng", "cmn")
        if isinstance(languages, str):
            return (languages.strip().lower(),)
        return tuple(str(x).strip().lower() for x in languages)

    @staticmethod
    def _build_lexicon_long(synset_words_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize lexical mappings to long format: synsetid, lang, lemma.

        Supported input schema:
          synsetid, lang, lemma
        """
        cols = set(synset_words_df.columns)
        if {"synsetid", "lang", "lemma"}.issubset(cols):
            out = synset_words_df[["synsetid", "lang", "lemma"]].copy()
            out = out[out["lemma"].notna()].reset_index(drop=True)
            out["lang"] = out["lang"].astype(str).str.lower()
            out["lemma"] = out["lemma"].astype(str)
            return out

        raise ValueError("synset_words_df must contain columns ('synsetid','lang','lemma').")

    def _resolve_language_selection(self, language) -> list[str]:
        """
        Resolve legacy and modern language selectors into OMW language codes.
        """
        if language is None:
            return list(self.available_languages)

        if isinstance(language, str):
            key = language.strip().lower()
            if key == "both":
                return list(self.available_languages)
            if key == "en":
                key = "eng"
            elif key == "zh":
                key = "cmn"
            return [key]

        out = []
        for item in language:
            key = str(item).strip().lower()
            if key == "en":
                key = "eng"
            elif key == "zh":
                key = "cmn"
            out.append(key)
        return out

    @staticmethod
    def _synset_to_id(syn) -> int:
        """
        Convert NLTK synset to SQL-style synset id:
        n -> 1XXXXXXXX, v -> 2XXXXXXXX, a/s -> 3XXXXXXXX, r -> 4XXXXXXXX
        """
        pos_prefix = {"n": 1, "v": 2, "a": 3, "s": 3, "r": 4}[syn.pos()]
        return pos_prefix * 100_000_000 + syn.offset()

    @classmethod
    def build_synset_words_from_nltk(cls, languages: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """
        Build multilingual synset_words DataFrame from NLTK WordNet + OMW lemmas.

        Returns long format columns: synsetid, lang, lemma.
        """
        if wn is None:
            raise ImportError("nltk.wordnet is not available. Install nltk and WordNet corpora.")

        if languages is None:
            languages = ("eng", "cmn")

        rows = []
        for syn in wn.all_synsets():
            sid = cls._synset_to_id(syn)
            for lang in languages:
                lemmas = syn.lemma_names(lang) or []
                for lemma in lemmas:
                    rows.append(
                        {
                            "synsetid": sid,
                            "lang": str(lang).lower(),
                            "lemma": str(lemma).replace("_", " "),
                        }
                    )

        if not rows:
            return pd.DataFrame(columns=["synsetid", "lang", "lemma"])

        return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)

    @classmethod
    def build_semlinks_from_nltk(cls) -> pd.DataFrame:
        """
        Build semlinks DataFrame from NLTK WordNet relations.
        """
        if wn is None:
            raise ImportError("nltk.wordnet is not available. Install nltk and WordNet corpora.")

        relation_methods = {
            "hypernym": "hypernyms",
            "hyponym": "hyponyms",
            "instance_hypernym": "instance_hypernyms",
            "instance_hyponym": "instance_hyponyms",
            "member_holonym": "member_holonyms",
            "substance_holonym": "substance_holonyms",
            "part_holonym": "part_holonyms",
            "member_meronym": "member_meronyms",
            "substance_meronym": "substance_meronyms",
            "part_meronym": "part_meronyms",
            "topic_domain": "topic_domains",
            "region_domain": "region_domains",
            "usage_domain": "usage_domains",
            "attribute": "attributes",
            "entailment": "entailments",
            "cause": "causes",
            "also_see": "also_sees",
            "verb_group": "verb_groups",
            "similar_to": "similar_tos",
        }

        rows = []
        for syn in wn.all_synsets():
            src = cls._synset_to_id(syn)
            for rel_name, method_name in relation_methods.items():
                neighbors = getattr(syn, method_name)()
                for nbr in neighbors:
                    rows.append(
                        {
                            "synset1id": src,
                            "synset2id": cls._synset_to_id(nbr),
                            "link": rel_name,
                        }
                    )

        if not rows:
            return pd.DataFrame(columns=["synset1id", "synset2id", "link"])

        return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)

    def _candidate_lemmas(self, word: str) -> set[str]:
        """
        Return candidate base forms for a seed word.

        Uses WordNet morphology when NLTK + corpus data are available.
        Falls back to the lowercased surface form.
        """
        w = str(word).strip().lower()
        if not w:
            return set()

        candidates = {w}
        if wn is None:
            return candidates

        try:
            for pos in (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV):
                lemma = wn.morphy(w, pos)
                if lemma:
                    candidates.add(lemma.lower())
        except Exception:
            # Keep behavior stable if nltk corpora are not available.
            return candidates

        return candidates

    def _heuristic_base_forms(self, word: str) -> set[str]:
        """
        Cheap morphology fallback when WordNet lemmatization is unavailable.
        """
        w = str(word).strip().lower()
        if not w:
            return set()

        forms = {w}
        if len(w) > 4 and w.endswith("ies"):
            forms.add(w[:-3] + "y")
        if len(w) > 3 and w.endswith("s"):
            forms.add(w[:-1])
        if len(w) > 5 and w.endswith("ing"):
            root = w[:-3]
            forms.add(root)
            forms.add(root + "e")
        if len(w) > 4 and w.endswith("ed"):
            root = w[:-2]
            forms.add(root)
            forms.add(root + "e")
        return forms

    def _expand_seed_words(
        self,
        seed_words: Sequence[str],
        use_lemmatization: bool = True,
    ) -> tuple[set[str], dict[str, set[str]]]:
        """
        Expand seed words to include candidate lemmas.

        Returns:
            expanded_words: all normalized forms used for lexical matching
            per_seed_forms: mapping from original seed -> normalized forms
        """
        expanded_words: set[str] = set()
        per_seed_forms: dict[str, set[str]] = {}
        for w in seed_words:
            w_norm = str(w).strip().lower()
            if not w_norm:
                continue

            # Strict mode: exact lexical match only (case-insensitive).
            if not use_lemmatization:
                forms = {w_norm}
            else:
                forms = self._heuristic_base_forms(w_norm)
                if use_lemmatization:
                    forms |= self._candidate_lemmas(w_norm)
            if forms:
                per_seed_forms[str(w)] = forms
                expanded_words |= forms
        return expanded_words, per_seed_forms


    def _build_graph(self):
        """
        Build an adjacency list from semantic links for synset graph traversal.
        """
        g = defaultdict(list)
        for row in self.semlinks.itertuples(index=False):
            g[row.synset1id].append(
                (row.synset2id, row.link)
            )
        self.graph = g

    def find_linked_synsets(
        self,
        seed_words: Sequence[str],
        max_depth: int = 2, # set to 0 if you want just the inputted words' synsets
        allowed_relations: Optional[set[str]] = None,
        max_degree: int = 30,
        decay: float = 0.7,
        seed_languages = None,
        use_lemmatization: bool = True,
        return_results: bool = False):
        """
        Expand seed words into related synsets using graph traversal and score them.
    
        Maps seed words to synsets, propagates through the synset graph up to max_depth,
        and assigns each synset a score based on lexical matches and distance-weighted reach.
    
        Sets:
            self.expanded_synsets: scored synsets
            self.selected_synsets: expanded synsets with embeddings
            self.missing_seed_words: seed words not found in WordNet

        Args:
            seed_languages: OMW language code(s) used to map seeds to synsets.
                Examples: "eng", "cmn", ["eng", "spa"], or None for all available.
            use_lemmatization: expand seed words with lemma candidates (WordNet + heuristics).
        """
        

        if not (0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1].")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0.")
        if allowed_relations is None:
            allowed_relations = set()
    
        # Lexical mapping: seed_words -> synsets, with counts (lexical hits)
        selected_langs = self._resolve_language_selection(seed_languages)
        df = self.lexicon_long[self.lexicon_long["lang"].isin(selected_langs)].copy()
        if df.empty:
            raise ValueError(f"No lexical rows found for seed_languages={selected_langs}")

        lemma_norm = df["lemma"].astype(str).str.strip().str.lower()
        expanded_words, per_seed_forms = self._expand_seed_words(
            seed_words,
            use_lemmatization=use_lemmatization,
        )
        if not expanded_words:
            raise ValueError("None of the provided seed words were recognized")
    
        mask = pd.Series(False, index=df.index)
        matched_seed_words = set()
    
        mask |= lemma_norm.isin(expanded_words)

        matched_rows = df.loc[mask, ["lemma"]].copy()
        matched_forms = set(matched_rows["lemma"].dropna().astype(str).str.strip().str.lower())

        for seed, forms in per_seed_forms.items():
            if forms & matched_forms:
                matched_seed_words.add(seed)
        self.missing_seed_words = sorted(set(map(str, seed_words)) - matched_seed_words)
    
        seed_synsets_series = df.loc[mask, "synsetid"]
        if seed_synsets_series.empty:
            raise ValueError("None of the provided seed words were recognized")

        self.seed_synsets = seed_synsets_series
            
        lexical_hits = Counter(seed_synsets_series.tolist())  # counts across seed_words matches (with duplicates in df)
        seed_synsets = list(lexical_hits.keys())
        
        # Graph propagation: BFS from each seed synset, count each reached synset once per seed synset
        reach_score = defaultdict(float)
        min_depth = {}  # minimum depth reached over all seeds
    
        self.reached_by_seeds: Dict[int, set[int]] = defaultdict(set)
    
        for s0 in seed_synsets:
            q = deque([(s0, 0)])
            seen_local = {s0: 0} 
    
            while q:
                s, d = q.popleft()
                if d > max_depth:
                    continue
    
                reach_score[s] += decay ** d
                if s not in min_depth or d < min_depth[s]:
                    min_depth[s] = d
                self.reached_by_seeds[s].add(s0)
    
                if d == max_depth:
                    continue
    
                neighbors = self.graph.get(s, [])
                if len(neighbors) > max_degree:
                    continue  # too general; do not expand further from here
    
                for nbr, rel in neighbors:
                    if allowed_relations and rel not in allowed_relations:
                        continue
                    nd = d + 1
                    if nbr not in seen_local:
                        seen_local[nbr] = nd
                        q.append((nbr, nd))
    
        #  Combine into a DataFrame
        all_synsets = set(reach_score.keys()) | set(lexical_hits.keys())
    
        rows = []
        for sid in all_synsets:
            lex = int(lexical_hits.get(sid, 0))
            r = float(reach_score.get(sid, 0.0))
            rows.append(
                {
                    "synsetid": sid,
                    "lexical_hits": lex,
                    "reach_score": r,
                    "score": lex + r,
                    "min_depth": int(min_depth.get(sid, np.nan)) if sid in min_depth else np.nan,
                    **({"reached_by_seeds": sorted(self.reached_by_seeds[sid])}),
                }
            )
    
        self.expanded_synsets = (pd.DataFrame(rows)
                                .sort_values(["score", "lexical_hits", "reach_score"], ascending=[False, False, False])
                                .reset_index(drop=True))
            
        self.selected_synsets = self.synset_embeddings.merge(self.expanded_synsets[['synsetid', 'score']], on='synsetid', how='inner')
    
        if return_results:
            return self.expanded_synsets

    def get_centroids(
        self,
        strategy = 'hdbscan',
        params = None
    ):
        """
        Returns a set of embeddings that will represent your topic.

        Args:
            strategy = Chooses the method used to define the topic embedding(s).
                mean = Returns the mean of the seed synset embeddings.
                weighted_mean = Returns the mean of the seed synset embeddings weighted by the topic score.
                identity = Just fetches and returns the embeddings for every seed synset.
                anchors = Grabs either the top m synsets by score or as many synsets as need to cover the mass fraction of total score and returns their embeddings.
                kmeans = Uses kmeans and silhouette scoring to discover topic clusters. Returns the centroids of discovered clusters.
                hdbscan = Uses hdbscan to discover topic clusters. Returns the centroids of discovered clusters.
            params = Takes a dictionary of parameters to be passed to some of the methods.
        Sets:
            self.centroids_df: centroid_id, embedding, score
        """
            
        params = params or {}
    
        allowed = {
            "mean": set(),
            "weighted_mean": set(),
            "identity": set(),
            "kmeans": {"n_clusters", "k_min", "k_max", "random_state"},
            "hdbscan": {"min_cluster_size", "min_samples", "epsilon", "method"},
            "anchors": {"m", "mass"},
        }
    
        if strategy not in allowed:
            raise ValueError(f"{strategy} is not a valid strategy.")
    
        unknown = set(params) - allowed[strategy]
        if unknown:
            raise ValueError(f"Unknown parameter(s) for {strategy}: {sorted(unknown)}")
    
        df = self.selected_synsets[["synsetid", "embedding", "score"]].copy()
        X = np.vstack(df["embedding"].values).astype(np.float32)
        scores = df["score"].astype(float).to_numpy()
    
        if strategy == "mean":
            c = X.mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-12)
            self.centroids_df =  pd.DataFrame({"centroid_id": [0], "embedding": [c], "score": [1.0]})
    
        elif strategy == "weighted_mean":
            w = np.clip(scores, 0, None)
            if w.sum() == 0:
                w = np.ones_like(w)
            c = np.average(X, axis=0, weights=w)
            c = c / (np.linalg.norm(c) + 1e-12)
            self.centroids_df =  pd.DataFrame({"centroid_id": [0], "embedding": [c], "score": [1.0]})
    
        elif strategy == "identity":
            emb = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            out = pd.DataFrame(
                {
                    "centroid_id": df["synsetid"].to_numpy(),
                    "embedding": list(emb),
                    "score": scores,
                }
            )
            self.centroids_df =  out.reset_index(drop=True)


        elif strategy == "anchors":
            m = params.get("m", None)
            mass = params.get("mass", None)
        
            df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)
        
            if m is not None:
                m = int(max(1, min(m, len(df_sorted))))
                anchors_df = df_sorted.iloc[:m]
        
            elif mass is not None:
                mass = float(mass)
                if not (0 < mass <= 1.0):
                    raise ValueError("mass must be in (0, 1].")
        
                total = df_sorted["score"].sum()
                if total <= 0:
                    anchors_df = df_sorted.iloc[:1]
                else:
                    cum = df_sorted["score"].cumsum() / total
                    anchors_df = df_sorted.loc[cum <= mass]
                    if anchors_df.empty:
                        anchors_df = df_sorted.iloc[:1]
        
            else:
                anchors_df = df_sorted.iloc[:3]
        
            Xc = np.vstack(anchors_df["embedding"].values).astype(np.float32)
            Xc = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
        
            self.centroids_df = pd.DataFrame(
                {
                    "centroid_id": anchors_df["synsetid"].to_numpy(),
                    "embedding": list(Xc),
                    "score": anchors_df["score"].astype(float).to_numpy(),
                }
            ).reset_index(drop=True)

        
        elif strategy == "kmeans":
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
    
            n_clusters = params.get("n_clusters", None)
            k_min = int(params.get("k_min", 2))
            k_max = int(params.get("k_max", 10))
            random_state = int(params.get("random_state", 0))
    
            if n_clusters is not None:
                k = int(n_clusters)
                k = max(1, min(k, len(df)))
                labels = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit_predict(X)
            else:
                k_min_eff = max(k_min, 2)
                k_max_eff = min(k_max, len(df) - 1)
    
                if k_max_eff < k_min_eff:
                    labels = np.zeros(len(df), dtype=int)
                else:
                    best_s, best_labels = -1.0, None
                    for k in range(k_min_eff, k_max_eff + 1):
                        lab = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit_predict(X)
                        if len(np.unique(lab)) < 2:
                            continue
                        s = silhouette_score(X, lab, metric="euclidean")
                        if s > best_s:
                            best_s, best_labels = s, lab
                    labels = best_labels if best_labels is not None else np.zeros(len(df), dtype=int)
    
            out_rows = []
            for cl in np.unique(labels):
                idx = np.where(labels == cl)[0]
                w = np.clip(scores[idx], 0, None)
                if w.sum() == 0:
                    w = np.ones_like(w)
                c = np.average(X[idx], axis=0, weights=w)
                c = c / (np.linalg.norm(c) + 1e-12)
                out_rows.append(
                    {"centroid_id": int(cl), "embedding": c.astype(np.float32), "score": float(w.mean())}
                )
    
            self.centroids_df =  pd.DataFrame(out_rows).sort_values("score", ascending=False).reset_index(drop=True)
    
        elif strategy == "hdbscan":
            import hdbscan
    
            min_cluster_size = int(params.get("min_cluster_size", 10))
            min_samples = params.get("min_samples", None)
            min_samples = None if min_samples is None else int(min_samples)
    
            epsilon = params.get("epsilon", None)
            epsilon = 0.0 if epsilon is None else float(epsilon)
    
            method = params.get("method", "eom")
    
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method=method,
                cluster_selection_epsilon=epsilon,
            )
            labels = clusterer.fit_predict(X)
    
            clusters = [c for c in np.unique(labels) if c != -1]
            if not clusters: # defaults to weighted mean
                w = np.clip(scores, 0, None)
                if w.sum() == 0:
                    w = np.ones_like(w)
                c = np.average(X, axis=0, weights=w)
                c = c / (np.linalg.norm(c) + 1e-12)
                self.centroids_df = pd.DataFrame({"centroid_id": [0], "embedding": [c], "score": [float(w.mean())]})

                return
    
            out_rows = []
            for cl in clusters:
                idx = np.where(labels == cl)[0]
                w = np.clip(scores[idx], 0, None)
                if w.sum() == 0:
                    w = np.ones_like(w)
                c = np.average(X[idx], axis=0, weights=w)
                c = c / (np.linalg.norm(c) + 1e-12)
                out_rows.append(
                    {"centroid_id": int(cl), "embedding": c.astype(np.float32), "score": float(w.mean())}
                )
    
            self.centroids_df = pd.DataFrame(out_rows).sort_values("score", ascending=False).reset_index(drop=True)

        else:
            raise ValueError(f"{strategy} is not a valid strategy.")
    
    def find_similar_words(
        self,
        min_similarity=0.65,
        scale_threshold_by_score=False,
        alpha=0.10,
        language = "both",
        word_col="word"):
        """
        Find synsets and words similar to topic centroids by cosine similarity.
    
        Compares each centroid against all synset embeddings, keeps synsets above
        a similarity threshold (optionally scaled by centroid score), and returns
        associated words.
    
        Returns:
          DataFrame with synset, centroid assignment, similarity, and word.
        """
        
        C = np.vstack(self.centroids_df["embedding"].values).astype(np.float32)
        cid = self.centroids_df["centroid_id"].to_numpy()
        cscore = self.centroids_df["score"].astype(float).to_numpy()
    
        X = np.vstack(self.synset_embeddings["embedding"].values).astype(np.float32)
        sid = self.synset_embeddings["synsetid"].to_numpy()
    
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    
        if scale_threshold_by_score:
            smin, smax = float(np.min(cscore)), float(np.max(cscore))
            if smax > smin:
                sn = (cscore - smin) / (smax - smin)
            else:
                sn = np.ones_like(cscore)
            thr = np.clip(min_similarity + alpha * (1.0 - sn), 0.0, 1.0)
        else:
            thr = np.full(len(cscore), float(min_similarity), dtype=float)
    
        rows = []
        for i in range(C.shape[0]):
            sims = X @ C[i]
            mask = sims >= thr[i]
            if not np.any(mask):
                continue
    
            rows.append(
                pd.DataFrame(
                    {
                        "synsetid": sid[mask],
                        "centroid_id": cid[i],
                        "centroid_score": cscore[i],
                        "similarity": sims[mask].astype(float),
                    }
                )
            )
    
        if not rows:
            return pd.DataFrame(columns=["synsetid", "centroid_id", "centroid_score", "similarity", word_col])
    
        matches = pd.concat(rows, ignore_index=True)
    
        matches = (
            matches.sort_values(["synsetid", "similarity"], ascending=[True, False])
                   .drop_duplicates("synsetid", keep="first")
                   .reset_index(drop=True)
        )
    
        selected_langs = self._resolve_language_selection(language)
        w_long = self.lexicon_long[self.lexicon_long["lang"].isin(selected_langs)][["synsetid", "lemma"]].copy()
        if w_long.empty:
            raise ValueError(f"No lexical rows found for language={selected_langs}")
        w_long = w_long.rename(columns={"lemma": word_col})
        w_long = w_long[w_long[word_col].notna()]
        w_long[word_col] = w_long[word_col].astype(str)
    
        out = matches.merge(w_long, on="synsetid", how="inner")
        return out[["synsetid", "centroid_id", "centroid_score", "similarity", word_col]].reset_index(drop=True)

    def top_k_words(self, df, k: int, word_col: str = "word"):
        """
        Select a global top-k list of words from centroid-expanded results.
    
        Allocates the word budget across centroids in proportion to their scores,
        then selects the highest-similarity words per centroid with global deduping.
        """
        k = int(k)
        if k <= 0 or df.empty:
            return pd.DataFrame(columns=["centroid_id", "centroid_score", "similarity", word_col])
    
        d = df[["centroid_id", "centroid_score", "similarity", word_col]].copy()
    
        d = (
            d.sort_values(["centroid_id", "similarity"], ascending=[True, False])
             .drop_duplicates(["centroid_id", word_col], keep="first")
             .reset_index(drop=True)
        )
    
        scores = (
            d[["centroid_id", "centroid_score"]]
            .drop_duplicates("centroid_id")
            .set_index("centroid_id")["centroid_score"]
            .astype(float)
            .clip(lower=0)
        )
        if scores.sum() <= 0:
            scores[:] = 1.0
    
        props = scores / scores.sum()
        raw = props * k
        base = np.floor(raw).astype(int)
        rem = k - int(base.sum())
    
        frac = (raw - base).sort_values(ascending=False)
        if rem > 0:
            for cid in frac.index[:rem]:
                base.loc[cid] += 1
        elif rem < 0:
            for cid in frac.sort_values().index[:(-rem)]:
                if base.loc[cid] > 0:
                    base.loc[cid] -= 1
    
        picks = []
        for cid, q in base.items():
            if q <= 0:
                continue
            sub = d[d["centroid_id"] == cid].head(int(q))
            if not sub.empty:
                picks.append(sub)
    
        out = pd.concat(picks, ignore_index=True) if picks else d.iloc[0:0].copy()
    
        out = (
            out.sort_values("similarity", ascending=False)
               .drop_duplicates(word_col, keep="first")
               .reset_index(drop=True)
        )
    
        if len(out) < k:
            remaining = d[~d[word_col].isin(out[word_col])]
            remaining = (
                remaining.sort_values("similarity", ascending=False)
                         .drop_duplicates(word_col, keep="first")
            )
            need = k - len(out)
            out = pd.concat([out, remaining.head(need)], ignore_index=True)
    
        out = out.sort_values("similarity", ascending=False).head(k).reset_index(drop=True)
        return out[["centroid_id", "centroid_score", "similarity", word_col]]
