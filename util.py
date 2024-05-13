import string
import nltk
from nltk.util import ngrams

# Custom stops words, built upon gensim.stopwords
stop_words = ({'want','diagnose', 'diagnosis','sick','illness','i','those', 'on', 'own', 'yourselves', 'ie', 'around', 'between', 'four', 'been', 'alone', 'off', 'am', 'then', 'other', 'can', 'cry', 'regarding', 'hereafter', 'front', 'too', 'used', 'wherein', 'doing', 'everything', 'up', 'never', 'onto', 'how', 'either', 'before', 'anyway', 'since', 'through', 'amount', 'now', 'he', 'cant', 'was', 'con', 'have', 'into', 'because', 'inc', 'not', 'therefore', 'they', 'even', 'whom', 'it', 'see', 'somewhere', 'interest', 'thereupon', 'thick', 'nothing', 'whereas', 'much', 'whenever', 'find', 'seem', 'until', 'whereby', 'at', 'ltd', 'fire', 'also', 'some', 'last', 'than', 'get', 'already', 'our', 'doesn', 'once', 'will', 'noone', 'that', 'what', 'thus', 'no', 'myself', 'out', 'next', 'whatever', 'although', 'though', 'etc', 'which', 'would', 'therein', 'nor', 'somehow', 'whereupon', 'besides', 'whoever', 'thin', 'ourselves', 'few', 'did', 'third', 'without', 'twelve', 'anything', 'against', 'while', 'twenty', 'if', 'however', 'found', 'herself', 'when', 'may', 'six', 'ours', 'done', 'seems', 'else', 'call', 'perhaps', 'had', 'nevertheless', 'fill', 'where', 'otherwise', 'still', 'within', 'its', 'for', 'together', 'elsewhere', 'throughout', 'of', 'eg', 'others', 'show', 'sincere', 'anywhere', 'anyhow', 'as', 'are', 'the', 'hence', 'something', 'hereby', 'nowhere', 'latterly', 'de', 'say', 'does', 'neither', 'his', 'go', 'forty', 'put', 'their', 'by', 'namely', 'km', 'could', 'five', 'unless', 'itself', 'is', 'nine', 'whereafter', 'down', 'bottom', 'thereby', 'such', 'both', 'she', 'become', 'whole', 'who', 'yourself', 'every', 'thru', 'except', 'very', 'several', 'among', 'being', 'be', 'mine', 'further', 'here', 'during', 'why', 'with', 'just', 'becomes', 'about', 'a', 'co', 'using', 'seeming', 'due', 'wherever', 'beforehand', 'detail', 'fifty', 'becoming', 'might', 'amongst', 'my', 'empty', 'thence', 'thereafter', 'almost', 'least', 'someone', 'often', 'from', 'keep', 'him', 'or', 'top', 'her', 'didn', 'nobody', 'sometime', 'across', 'hundred', 'only', 'via', 'name', 'eight', 'three', 'back', 'to', 'all', 'became', 'move', 'me', 'we', 'formerly', 'so', 'i', 'whence', 'describe', 'under', 'always', 'himself', 'more', 'herein', 'in', 'after', 'themselves', 'you', 'them', 'above', 'sixty', 'hasnt', 'your', 'made', 'everywhere', 'indeed', 'most', 'kg', 'fifteen', 'but', 'must', 'along', 'beside', 'hers', 'computer', 'side', 'former', 'full', 'anyone', 'has', 'yours', 'whose', 'behind', 'please', 'mill', 'amoungst', 'ten', 'seemed', 'sometimes', 'should', 'over', 'take', 'each', 'don', 'same', 'rather', 'really', 'latter', 'and', 'part', 'hereupon', 'per', 'eleven', 'ever', 'enough', 'again', 'us', 'yet', 'moreover', 'mostly', 'one', 'meanwhile', 'whither', 'there', 'toward', 'give', 'system', 'do', 'quite', 'an', 'these', 'everyone', 'towards', 'this', 'bill', 'cannot', 'un', 'afterwards', 'beyond', 'make', 'were', 'whether', 'well', 'another', 'below', 'first', 'upon', 'any', 'none', 'many', 'various', 'serious', 're', 'two', 'less', 'couldnt'})
def preprocess_symptoms(text):
    """
    Generates ngrams of KEYWORDS in the format of the column titles, free of punctuation and stop words.
        :param text (str): The text to preprocess.
        :return: tokens, A list of preprocessed tokens.
    """
    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]
    ngrams_list = []
    for n in range(1, 4):  # Generate 1-grams, 2-grams, and 3-grams
        ngrams1 = ngrams(tokens, n)
        ngrams_list.extend(["_".join(gram) for gram in ngrams1])

    return ngrams_list

# print(preprocess("i have a runny nose and i am feeling all kinds of bad, a sore throat too"))


def download_nltk_resources():
    # Download resources if necessary
    # Try except : catch the errors rather than print to the user
    resources = {
        "tokenizers/punkt": "punkt",
        "corpora/wordnet": "wordnet",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
        "corpora/stopwords": "stopwords"
    }
    for resource_path, resource_id in resources.items():
        try:
            # Check if the resource is available, and download if it is not
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource: {resource_id}")
            nltk.download(resource_id)


def set_name(my_input):
    # TODO: If u type hello, it will take this as the name, or just the first word!!! fix? ne_chunks? ngrams? NER (look for (name, NNP) tuples
    # Use library to perform entity recognition using bag of words tagging
    name = "NOT FOUND"

    # Pinpoint which word is the name in input:
    pos_tags = nltk.pos_tag(nltk.word_tokenize(my_input))
    for entity in pos_tags:
        if isinstance(entity, tuple) and entity[1] == 'NNP':  # NNP: Proper noun, singular
            name = entity[0]
            # print("NNP Name found: %s" % entity[0])
            # Once ideal name is found, the system stops looking
            return name
        elif isinstance(entity, tuple) and entity[1] == 'NN':
            name = entity[0]
            # print("Name found: %s" % entity[0])
        elif len(pos_tags) == 1:
            name = entity[0]
    return name