BREAK_ON_ASSERT = True

def ASSERT(pred):
    if BREAK_ON_ASSERT:
        if not pred:
            import ipdb; ipdb.set_trace()
    else:
        assert pred
