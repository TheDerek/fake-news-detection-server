from flask import Flask, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from newspaper import Article

from . import Predictor

app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["400 per day", "200 per hour"]
)

print('Loading the predictor and associated TensorFlow things')
predictor = Predictor()


@app.route('/get/<path:url>')
def get(url):
    app.logger.info('Fetching article with URL: {}'.format(url))
    article = Article(url)
    article.download()
    article.parse()
    document = article.text
    app.logger.info('Parsing article with URL: {}'.format(url))

    app.logger.info('Article text:\n{}'.format(document))

    return jsonify({
        'is_article': True,
        'category': predictor.predict(document)
    }), 200