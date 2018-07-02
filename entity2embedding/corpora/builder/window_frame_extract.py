import collections


def window_frame_extract(record_list, skip_window):
    target_buffer = []
    context_buffer = []

    # padding = [None] * skip_window
    # line_with_padding = padding + record_list + padding

    padding = [None] * skip_window
    line_with_padding = padding + record_list + padding

    # window => [ skip_window target skip_window ]
    window_span = 2 * skip_window + 1

    window = collections.deque(maxlen=window_span)

    # fulfill the window
    window.extend(line_with_padding[:window_span])
    word_index = window_span

    center_index_of_window = skip_window

    while True:
        context_range = list(range(0, skip_window)) \
                      + list(range(skip_window + 1, window_span))
        for i in context_range:
            context_word = window[i]

            # skip None
            if context_word is None:
                continue

            target_buffer.append(window[center_index_of_window])
            context_buffer.append(context_word)

        # update the window or exit
        try:
            next_word = line_with_padding[word_index]
        except IndexError:
            # reach the EOL
            break
        else:
            window.append(next_word)
            word_index += 1

    return list(zip(target_buffer, context_buffer))


if __name__ == "__main__":
    result = window_frame_extract(["王小明", "在", "北京", "的", "清华大学", "读书"], 1)
    print(result)
