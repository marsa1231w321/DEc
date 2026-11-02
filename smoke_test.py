import dec_engine
from dec_engine.app import build_demo_scene

print(dec_engine.__version__, len(build_demo_scene().objects))
