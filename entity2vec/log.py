LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s # %(levelname)s @ %(module)s [%(process)d | %(thread)d] : %(message)s'
        },
        'simple': {
            'format': '%(levelname)s %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': './log.log',
            'formatter': 'verbose'
        },
    },
    'loggers': {
        'tensorflow_word2vec': {
            'handlers': ['file'],
            'propagate': True,
            'level': 'DEBUG',
        }
    }
}
