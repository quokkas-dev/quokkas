

def str_of_args(args, kwargs):
    """
    Gets a string representation of provided args and kwargs

    :param args: iterable of arguments
    :param kwargs: dictionary of keyword arguments
    :return: string representation
    """

    string_arr = []
    for i in args:
        if hasattr(i, 'shape'):  # numpy array, dataframe, tensor
            string_arr.append(i.__class__.__name__ + ', ')
        else:
            string_arr.append(repr(i) + ', ')
    for i in kwargs:
        if hasattr(i, 'shape'):  # numpy array, dataframe, tensor
            string_arr.append(i.__class__.__name__ + ', ')
        else:
            string_arr.append(f"{i} = {str(kwargs[i])}, ")
    return ''.join(string_arr)[:-2]
