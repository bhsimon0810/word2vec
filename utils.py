def read_lines_from_file_as_data_chunks(file_name, chunk_size, callback, return_whole_chunk=False):
    """read file line by line regardless of its size

    Args:
        file_name: absolute path of file to read
        chunk_size: size of data to be read at at time
        callback: callback method, prototype ----> def callback(data, eof, file_name)
    """

    def read_in_chunks(file_obj, chunk_size=5000):
        """
        https://stackoverflow.com/a/519653/5130720
        Lazy function to read a file
        Default chunk size: 5000.
        """
        while True:
            data = file_obj.read(chunk_size)
            if not data:
                break
            yield data

    fp = open(file_name, encoding='utf-8')
    data_left_over = None

    # loop through characters
    for chunk in read_in_chunks(fp, chunk_size):

        # if uncompleted data exists
        if data_left_over:
            # print('\n left over found')
            current_chunk = data_left_over + chunk
        else:
            current_chunk = chunk

        # split chunk by new line
        lines = current_chunk.splitlines()

        # check if line is complete
        if current_chunk.endswith('\n'):
            data_left_over = None

        else:
            data_left_over = lines.pop()

        if return_whole_chunk:
            callback(data=lines, eof=False, file_name=file_name)

        else:

            for line in lines:
                callback(data=line, eof=False, file_name=file_name)
                pass

    if data_left_over:

        current_chunk = data_left_over
        if current_chunk is not None:

            lines = current_chunk.splitlines()

            if return_whole_chunk:
                callback(data=lines, eof=False, file_name=file_name)

            else:
                for line in lines:
                    callback(data=line, eof=False, file_name=file_name)
                    pass

    callback(data=None, eof=True, file_name=file_name)
