# -*- coding: utf-8 -*-
from flask import Flask
app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return 'Index Page'

@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/projects/')
def projects():
    return 'The project page'

@app.route('/about')
def about():
    return 'The about page'

app.run()