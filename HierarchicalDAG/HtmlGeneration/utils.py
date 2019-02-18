import time
import sys

def print_out(msg, file=None, new_line=True):
    # write to stdout
    sys.stdout.write(msg)
    if new_line:
        sys.stdout.write("\n")
    sys.stdout.flush()

    # write to file if exists
    if file is not None:
        file.write(msg)
        if new_line:
            file.write("\n")

def set_print_out_default_file(file):
    new_line = print_out.__defaults__[1]
    print_out.__defaults__ = (file, new_line)

def func_timing_short(print_func=print):
    def arg_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # call original function to time
            func_ret = func(*args, **kwargs)

            end_time = time.time()
            print_func("Elapsed time for {}: {:,.3f}s".format(
                func.__name__, end_time - start_time))

            # return original function return value
            return func_ret
        return wrapper
    return arg_decorator

def func_timing_long(print_func=print):
    def arg_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_local_time = time.ctime(start_time)
            print_func("Start time: {}".format(start_local_time))

            # call original function to time
            func_ret = func(*args, **kwargs)

            end_time = time.time()
            end_local_time = time.ctime(end_time)
            print_func("End time: {}".format(end_local_time))

            elapsed_time = int(end_time - start_time)
            et_minutes, et_seconds = divmod(elapsed_time, 60)
            et_hours, et_minutes = divmod(et_minutes, 60)
            if et_hours > 0:
                print_func("Elapsed time for {}: {:02d}h {:02d}m {:02d}s".format(
                    func.__name__, et_hours, et_minutes, et_seconds))
            elif et_minutes > 0:
                print_func("Elapsed time for {}: {:02d}m {:02d}s".format(
                    func.__name__, et_minutes, et_seconds))
            else:
                print_func("Elapsed time for {}: {:02d}s".format(
                    func.__name__, et_seconds))

            # return original function return value
            return func_ret
        return wrapper
    return arg_decorator