import copy
from typing import Optional

from snorkel.map import BaseMapper, LambdaMapper, Mapper, lambda_mapper
from snorkel.types import DataPoint

"""Base classes for preprocessors.

A preprocessor is a data point to data point mapping in a labeling
pipeline. This allows Snorkel operations (e.g. LFs) to share common
preprocessing steps that make it easier to express labeling logic.
A simple example for text processing is concatenating the title and
body of an article. For a more complex example, see
``snorkel.preprocess.nlp.SpacyPreprocessor``.
"""

# Used for type checking only
# Note: subclassing as below trips up mypy
BasePreprocessor = BaseMapper


class Preprocessor(Mapper):
    """Base class for preprocessors.

    See ``snorkel.map.core.Mapper`` for details.
    """

    pass


class LambdaPreprocessor(LambdaMapper):
    """Convenience class for defining preprocessors from functions.

    See ``snorkel.map.core.LambdaMapper`` for details.
    """

    pass


class preprocessor(lambda_mapper):
    """Decorate functions to create preprocessors.

    See ``snorkel.map.core.lambda_mapper`` for details.

    Example
    -------
    >>> @preprocessor()
    ... def combine_text_preprocessor(x):
    ...     x.article = f"{x.title} {x.body}"
    ...     return x
    >>> from snorkel.preprocess.nlp import SpacyPreprocessor
    >>> spacy_preprocessor = SpacyPreprocessor("article", "article_parsed")

    We can now add our preprocessors to an LF.

    >>> preprocessors = [combine_text_preprocessor, spacy_preprocessor]
    >>> from snorkel.labeling.lf import labeling_function
    >>> @labeling_function(pre=preprocessors)
    ... def article_mentions_person(x):
    ...     for ent in x.article_parsed.ents:
    ...         if ent.label_ == "PERSON":
    ...             return ABSTAIN
    ...     return NEGATIVE
    """

    def __call__(self, x: DataPoint) -> Optional[DataPoint]:
        """Run mapping function on input data point.

        Deep copies the data point first so as not to make
        accidental in-place changes. If ``memoize`` is set to
        ``True``, an internal cache is checked for results. If
        no cached results are found, the computed results are
        added to the cache.

        Parameters
        ----------
        x
            Data point to run mapping function on

        Returns
        -------
        DataPoint
            Mapped data point of same format but possibly different fields
        """
        if self.memoize:
            # NB: don't do ``self._cache.get(...)`` first in case cached value is ``None``
            x_hashable = self._memoize_key(x)
            if x_hashable in self._cache:
                return self._cache[x_hashable]
        # Dangerous; avoids pickling but copy() 
        # isn't perfect in avoiding mutation on original object.
        x_mapped = copy.deepcopy(x)
        # x_mapped = pickle.loads(pickle.dumps(x))
        for mapper in self._pre:
            x_mapped = mapper(x_mapped)
        x_mapped = self._generate_mapped_data_point(x_mapped)
        if self.memoize:
            self._cache[x_hashable] = x_mapped
        return x_mapped

    def __repr__(self) -> str:
        pre_str = f", Pre: {self._pre}"
        return f"{type(self).__name__} {self.name}{pre_str}"

