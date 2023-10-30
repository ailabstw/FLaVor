import os

import EXAMPLE_MODEL

from flavor.serve.api import InferAPIApp

app = InferAPIApp(callback=EXAMPLE_MODEL)

app.run(port=int(os.getenv("PORT", 9000)))
