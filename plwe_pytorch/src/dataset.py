import numpy as np
import h5py

def read_h5_pos(file, pos, nsamples):
    h5file = h5py.File(file, mode='r')
    data = h5file['data'][pos:pos+nsamples]
    label = h5file['label'][pos:pos+nsamples]
    h5file.close()
    return data, label


def read_h5_length(file):
    h5file = h5py.File(file, mode='r')
    length = len(h5file['data'])
    h5file.close()
    return length


class HDF5_Dataset_transpose():
    def __init__(self, hdf5_list, batchsize):
        self.hdf5_list = hdf5_list
        self.nfiles = len(hdf5_list)
        self.batch_size = batchsize
        self.epoches = 0
        self.curr_file = 0
        self.curr_file_pointer = 0

        length_list = list(map(read_h5_length, hdf5_list))
        self.len_list = length_list
        self.total_count = np.sum(length_list)

    def __len__(self):
        return self.total_count

    def __iter__(self):
        return self

    def __next__(self):
        h5_file = self.hdf5_list[self.curr_file]
        data, label = read_h5_pos(h5_file, self.curr_file_pointer, self.batch_size)
        data = np.transpose(data, (0, 1, 3, 2))
        label = np.transpose(label, (0, 1, 3, 2))
        self.curr_file_pointer += self.batch_size
        if self.curr_file_pointer >= self.len_list[self.curr_file]:
            self.curr_file_pointer = 0
            self.curr_file += 1
            if self.curr_file + 1 > self.nfiles:
                self.curr_file = 0
                self.epoches += 1
        return data, label


class HDF5_Dataset_transpose_uint8(HDF5_Dataset_transpose):
    def __next__(self):
        h5_file = self.hdf5_list[self.curr_file]
        data, label = read_h5_pos(h5_file, self.curr_file_pointer, self.batch_size)
        data = np.transpose(data, (0, 1, 3, 2)).astype(np.float32) / 255.
        label = np.transpose(label, (0, 1, 3, 2)).astype(np.float32) / 255.
        self.curr_file_pointer += self.batch_size
        if self.curr_file_pointer >= self.len_list[self.curr_file]:
            self.curr_file_pointer = 0
            self.curr_file += 1
            if self.curr_file + 1 > self.nfiles:
                self.curr_file = 0
                self.epoches += 1
        return data, label


