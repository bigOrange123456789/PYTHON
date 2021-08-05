from flask import Flask, g,render_template, request, jsonify

app = Flask(__name__)

g.test=1;

@app.route('/')
#@app.route('/s')
def question_page():
    g.test=g.test+1;
    return "<script>alert("+g.test+")</script>"

app.run()