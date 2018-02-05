# coding=utf-8

import json
import threading
import tornado.ioloop
import tornado.web
from model import Model

model = Model()
class MainHandler(tornado.web.RequestHandler):

    def get(self):
        key = "text"
        if self.request.arguments.has_key(key):
            text = self.get_argument(key)
            self.process(text)
        else:
            self.set_status(500)
            self.write("empty arguments")
            self.finish()
            return

    def post(self):
        if (not self.request.body):
            self.set_status(500)
            self.write("empty request")
            self.finish()
            return
        try:
            self.process(self.request.body.decode("utf-8"))
            return
        except Exception as e:
            self.set_status(500)
            self.write("internal server error: %s" % e)
            self.finish()

    def process(self, text):
        print(text)
        response = model.eval(text)
        self.set_header("Content-Type", "application/octet-stream")
        self.write(json.dumps(response))
        self.finish()

def make_app():
    return tornado.web.Application([(r"/", MainHandler)])

if __name__ == "__main__":
    app = make_app()
    app.listen(80)
    print("running \n")
    tornado.ioloop.IOLoop.current().start()
