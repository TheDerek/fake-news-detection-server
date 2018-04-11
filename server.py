from flask import Flask, jsonify
from newspaper import Article

app = Flask(__name__)


@app.route("/get/<path:url>")
def get(url):
    app.logger.info('Fetching article with URL: {}'.format(url))
    article = Article(url)
    article.download()

    app.logger.info('Parsing article with URL: {}'.format(url))
    article.parse()

    app.logger.info('Article text:\n{}'.format(article.text))

    return jsonify({
        'is_article': True,
        'category': 'real'
    }), 200