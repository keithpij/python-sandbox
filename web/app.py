from flask import Flask
import os
import socket


app = Flask(__name__)
visits = 0


@app.route('/')
def hello_world():
    global visits
    visits += 1
    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>" \
           "<b>Visits:</b> {visits}"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname(), visits=visits)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
