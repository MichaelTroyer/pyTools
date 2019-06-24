def print_exception_full_stack(print_locals=True):
    """Print full stack in a more orderly way
       Optionally print the exception frame local variables"""
    exc = sys.exc_info()  # 3-tuple (type, value, traceback)
    if exc is None:
        return None

    tb_type, tb_value, tb_obj = exc[0], exc[1], exc[2]
    exc_type = str(tb_type).split(".")[1].replace("'>", '')
    lg.logging(1, '\n\n'+header+'\n'+header)
    lg.logging(1,'\nEXCEPTION:\n{}\n{}\n'.format(exc_type, tb_value))
    lg.logging(1, header+'\n'+header+'\n\n')
    lg.logging(1, 'Traceback (most recent call last):')

    # 4-tuple (filename, line no, func name, text)
    tb = traceback.extract_tb(exc[2])
    for tb_ in tb:
        lg.logging(1, "{}\n"
                   "Filename: {}\n"
                   "Line Number: {}\n"
                   "Function Name: {}\n"
                   "Text: {}\n"
                   "Exception: {}"
                   "".format(header, tb_[0], tb_[1], tb_[2],
                             textwrap.fill(tb_[3]), exc[1]))
    if print_locals:
        stack = []
        while tb_obj.tb_next:
            tb_obj = tb_obj.tb_next  # Make sure at end of stack
        f = tb_obj.tb_frame          # Get the frame object(s)

        while f:                     # Append and rewind, reverse order
            stack.append(f)
            f = f.f_back
        stack.reverse()

        lg.logging(3, '\n\nFrames and locals (innermost last):\n'+header)
        for frame in stack:
            if str(frame.f_code.co_filename).endswith(filename):
                lg.logging(3, "{}\n"
                           "FRAME {} IN:\n"
                           "{}\n"
                           "LINE: {}\n"
                           "".format(header,
                                     textwrap.fill(frame.f_code.co_name),
                                     textwrap.fill(frame.f_code.co_filename),
                                     frame.f_lineno))

                if not frame.f_locals.items():
                    lg.logging(3, "No locals\n")

                else:
                    lg.logging(3, "{} LOCALS:\n".format(frame.f_code.co_name))
                    for key, value in sorted(frame.f_locals.items()):
                        # Exclude private and the i/o and header parameters
                        if not str(key).startswith("_"):
                            if not str(key) in ['In', 'Out', 'header']:
                                lg.logging(3, (str(key)+":").strip())

                                try:
                                    lg.logging(3, str(value).strip()+'\n')
                                except:
                                    lg.logging(3, 'Error writing value')
    return
