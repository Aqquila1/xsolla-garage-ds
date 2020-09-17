# -*- coding: utf-8 -*-
from flask import Flask

application = Flask(__name__)

from app import routes
