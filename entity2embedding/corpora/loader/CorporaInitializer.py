from tensorflow.python.training import session_run_hook


class CorporaInitializer(session_run_hook.SessionRunHook):
    def __init__(self, corpora_object):
        self.corpora_object = corpora_object

    def after_create_session(self, session, coord):
        session.run(self.corpora_object.initializer())