import logging
import pandas as pd
import tqdm

from overrides import overrides

from allennlp.data import dataset
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers import WordTokenizer

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field
from allennlp.data.fields import LabelField
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token

from nltk.tree import Tree
from typing import Dict
from typing import List

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("SSTReader")
class SSTReader(DatasetReader):
    """
    Reads tokens and their sentiment labels from the Stanford Sentiment Treebank.
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    use_subtrees : ``bool``, optional, (default = ``False``)
        Whether or not to use sentiment-tagged subtrees.
    granularity : ``str``, optional (default = ``"5-class"``)
        One of ``"5-class"``, ``"3-class"``, or ``"2-class"``, indicating the number
        of sentiment labels to use.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be consumed lazily.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 use_subtrees: bool = False,
                 granularity: str = "5-class",
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(token_min_padding_length=5)}
        self._use_subtrees = use_subtrees
        allowed_granularities = ["5-class", "3-class", "2-class"]
        if granularity not in allowed_granularities:
            raise ConfigurationError("granularity is {}, but expected one of: {}".format(
                    granularity, allowed_granularities))
        self._granularity = granularity

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file.readlines():
                line = line.strip("\n")
                if not line:
                    continue
                parsed_line = Tree.fromstring(line)
                if self._use_subtrees:
                    for subtree in parsed_line.subtrees():
                        instance = self.text_to_instance(subtree.leaves(), subtree.label())
                        if instance is None:
                            continue
                        yield instance
                else:
                    instance = self.text_to_instance(parsed_line.leaves(), parsed_line.label())
                    if instance is None:
                        continue
                    yield instance

    @overrides
    def text_to_instance(self, tokens: List[str], sentiment: str = None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in a given sentence.
        sentiment ``str``, optional, (default = None).
            The sentiment for this sentence.
        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The sentiment label of the sentence or phrase.
        """
        # pylint: disable=arguments-differ
        text_field = TextField([Token(x) for x in tokens], token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
        if sentiment is not None:
            # Convert to 3-class.
            if self._granularity == "3-class":
                if int(sentiment) < 2:
                    sentiment = "0"
                elif int(sentiment) == 2:
                    sentiment = "1"
                else:
                    sentiment = "2"
            elif self._granularity == "2-class":
                if int(sentiment) < 2:
                    sentiment = "0"
                elif int(sentiment) == 2:
                    return None
                else:
                    sentiment = "1"
            fields['adv'] = LabelField(0, skip_indexing=True)
            fields['label'] = LabelField(int(sentiment), skip_indexing=True)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'StanfordSentimentTreeBankTokensDatasetReader':
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        use_subtrees = params.pop('use_subtrees', False)
        granularity = params.pop_choice('granularity', ["5-class", "3-class", "2-class"], True)
        lazy = params.pop('lazy', False)
        params.assert_empty(cls.__name__)
        return StanfordSentimentTreeBankTokensDatasetReader(
                token_indexers=token_indexers, use_subtrees=use_subtrees,
                granularity=granularity, lazy=lazy)

@DatasetReader.register('CSVReader')
class CSVReader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer = WordTokenizer(), lazy: bool = False, max_length: int = 500) -> None:
        super().__init__(lazy)   
        self._tokenizer = tokenizer
        self._token_indexer = {"tokens": SingleIdTokenIndexer(lowercase_tokens=True, token_min_padding_length=5)}
        self.max_length = max_length

    @overrides
    def _read(self, file: str):
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            text = row['text']
            if 'label' in df.columns:
                target = int(row['label'])
                yield self.text_to_instance(text, target)
            else:
                yield self.text_to_instance(text)
            
    @overrides
    def text_to_instance(self, text: str, target: float = None) -> Instance:
        text = self._tokenizer.tokenize(text)
        text = text[:self.max_length]
        fields = {}
        # fields['idx'] = MetadataField({'idx': idx})
        fields['adv'] = LabelField(0, skip_indexing=True)
        fields['tokens'] = TextField(text, self._token_indexer)
        if target is not None:
            fields['label'] = LabelField(target, skip_indexing=True)
        return Instance(fields)

