#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

import config  # only for N_fix_per_session, TODO: rewrite

# utilities
# -----

# custom functions
# -----


def show_batch(sample_batched):
    """
    sample_batched: Tuple[torch.tensor, torch.tensor] -> None
    show_batch takes a contrastive sample sample_batched and plots
    an overview of the batch
    """

    grid_border_size = 2
    nrow = 10

    batch_1 = sample_batched[0][0][:, 0:, :, :]
    batch_2 = sample_batched[0][1][:, 0:, :, :]
    difference = np.abs(batch_1 - batch_2)

    titles = ["first contrast", "second contrast", "difference"]

    fig, axes = plt.subplots(1, 3, figsize=(2 * 6.4, 4.8))
    for (i, batch) in enumerate([batch_1, batch_2, difference]):
        ax = axes[i]
        grid = utils.make_grid(batch, nrow=nrow, padding=grid_border_size)
        ax.imshow(grid.numpy().transpose((1, 2, 0)))
        ax.set_title(titles[i])
        ax.axis("off")
    plt.show()


# ----------------
# custom classes
# ----------------

# custom CLTT dataset superclass (abstract)
# -----

class CLTTDataset(Dataset):
    """
    CLTTDataset is an abstract class implementing all the necessary methods
    to sample data according to the CLTT approach. CLTTDataset itself
    should not be instantiated as a standalone class, but should be
    inherited from and abstract methods should be overwritten
    """

    def __init__(self, root, split='train', transform=None, target_transform=None,
                 n_fix=5, contrastive=True, sampling_mode='uniform',
                 shuffle_object_order=True, circular_sampling=True, buffer_size=12096):
        """
        __init__ initializes the CLTTDataset Class, it defines class-wide
        constants and builds the registry of files and the data buffer

        root:str path to the dataset directory
        split:str supports 'train', 'test', 'val'
        transform:torchvision.transform
        target_transform:torchvision.transform
        n_fix:int for deterministic n_fix, float for probabilistic
        contrastive:bool contrastive dataset mode
        sampling_mode:str how the buffer gets built
        circular_sampling:bool make the first object the last object
        buffer_size:int approximate buffersize

        """
        super().__init__()

        # check if split is string and it is one of the valid options
        valid_splits = ['train', 'test', 'val']
        # add alternative split with potential k crossfolds
        valid_splits += [v+f'_alt_{k}' for v in valid_splits for k in range(5)]
        assert isinstance(
            split, str) and split in valid_splits, f'variable split has to be of type str and one of the following, {valid_splits}'

        self.split = split
        self.sampling_mode = sampling_mode
        self.shuffle_object_order = shuffle_object_order
        self.buffer_size = buffer_size
        self.n_fix = n_fix
        self.tau_plus = 1
        self.tau_minus = 0  # contrasts from the past (experimental)
        self.objects_in_current_set = 'all'  # have all objects
        # available for training and testing by default

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.contrastive = contrastive
        self.circular_sampling = circular_sampling

        self.get_dataset_properties()

        self.registry = self.build_registry(split)

        if self.contrastive:
            self.buffer = self.build_buffer(
                self.registry, self.sampling_mode, self.n_fix, self.shuffle_object_order, approx_size=self.buffer_size)
        else:
            # if used in non-contrastive mode the sampler just samples from all data
            self.buffer = self.registry
        pass

    def __len__(self):
        """
        __len__ defines the length of the dataset and indirectly
        defines how many samples can be drawn from the dataset
        in one epoch
        """
        length = len(self.buffer)
        return length

    def get_dataset_properties(self):
        """
        get_dataset_properties has to be defined for each dataset
        it stores number of objects, number of classes, a list of
        strings with labels
        """

        # basic properties (need to be there)
        self.n_objects = 3  # number of different objects >= n_classes
        self.n_classes = 3  # number of different classes
        self.labels = [
            "A",
            "B",
            "C",
        ]
        self.n_views_per_object = 10  # how many overall views of each object
        self.subdirectory = '/dataset_name/'  # where is the dataset
        self.name = 'dataset name'  # name of the dataset

        # custom properties (optional, dataset specific)
        # (anything you would want to have available in self)
        self.custom_property = ['one', 'two', 'three']

        raise Exception("Calling abstract method, please inherit \
        from the CLTTDataset class and reimplement this method")  # pseudoclass
        pass

    def __getitem__(self, idx):
        """
        __getitem__ is a method that defines how one sample of the
        dataset is drawn
        """
        if self.contrastive:
            image, label = self.get_single_item(idx)
            augmentation, _ = self.sample_contrast(idx)

            if self.transform:
                image, augmentation = self.transform(
                    image), self.transform(augmentation)

            if self.target_transform:
                label = self.target_transform(label)

            output = ([image, augmentation], label)
        else:
            image, label = self.get_single_item(idx)

            if self.transform:
                image = self.transform(image)

            if self.target_transform:
                label = self.target_transform(label)

            output = image, label

        return output

    def sample_contrast(self, chosen_index):
        """
        given index chosen_index, sample a corresponding contrast close in time
        """
        chosen_time = self.buffer.iloc[chosen_index]["time_idx"]

        possible_indices = self.buffer[
            (self.buffer["time_idx"].between(chosen_time - self.tau_minus, chosen_time + self.tau_plus)) & (
                self.buffer["time_idx"] != chosen_time)].index

        # sampling at the end of the buffer
        if (chosen_time + self.tau_plus) > self.buffer.time_idx.max():
            if self.circular_sampling:
                also_possible = self.buffer[
                    (self.buffer["time_idx"].between(self.buffer.time_idx.min(), (
                        chosen_time + self.tau_plus - 1) - self.buffer.time_idx.max())) & (
                        self.buffer["time_idx"] != chosen_time)].index
            else:
                also_possible = self.buffer[self.buffer["time_idx"]
                                            == chosen_time].index

            possible_indices = possible_indices.union(also_possible)

        # sampling at the beginning of the buffer
        if (chosen_time - self.tau_minus) < self.buffer.time_idx.min():
            if self.circular_sampling:
                also_possible = self.buffer[
                    (self.buffer["time_idx"].between(self.buffer.time_idx.max() + (chosen_time - self.tau_minus) + 1,
                                                     self.buffer.time_idx.max())) & (
                        self.buffer["time_idx"] != chosen_time)].index
            else:
                also_possible = self.buffer[self.buffer["time_idx"]
                                            == chosen_time].index

            possible_indices = possible_indices.union(also_possible)

        chosen_index = np.random.choice(possible_indices)
        return self.get_single_item(chosen_index)

    def get_single_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif isinstance(idx, pd.core.indexes.numeric.Int64Index):
            idx = idx[0]

        path_to_file = self.buffer.loc[idx, "path_to_file"]
        if isinstance(path_to_file, pd.core.series.Series):
            path_to_file = path_to_file.item()

        image = Image.open(path_to_file)
        obj_info = self.buffer.iloc[idx, 1:].to_dict()

        label = self.buffer.loc[idx, "label"]
        return image, label

    def build_registry(self, split):
        """
        build a registry of all image files
        """
        path_list = []
        object_list = []
        label_list = []
        time_list = []

        if split == 'train':
            d = self.root + self.subdirectory + 'train/'
            assert os.path.isdir(d), 'Train directory does not exist'
        elif split == 'test':
            d = self.root + self.subdirectory + 'test/'
            assert os.path.isdir(d), 'Test directory does not exist'
        else:
            d = self.root + self.subdirectory + 'validation/'
            if not(os.path.isdir(d)):
                print(
                    '[INFO] Validation directory does not exist, using testset instead')
                d = self.root + self.subdirectory + 'test/'

        # have an ordered list
        list_of_files = os.listdir(d)
        list_of_files.sort()

        for timestep, path in enumerate(list_of_files):
            full_path = os.path.join(d, path)
            if os.path.isfile(full_path):
                path_list.append(full_path)
                object_list.append(timestep // self.n_views_per_object)
                label_list.append(timestep // self.n_views_per_object)
                time_list.append(timestep % self.n_views_per_object)

        tempdict = {'path_to_file': path_list, 'label': label_list,
                    'object_nr': object_list, 'time_idx': time_list}

        dataframe = pd.DataFrame(tempdict)
        dataframe.sort_values(by=['object_nr', 'time_idx'], inplace=True)
        dataframe.reset_index(drop=True, inplace=True)

        return dataframe

    def build_buffer(self, registry, sampling_mode, n_fix, shuffle_object_order, approx_size):
        """
        build_buffer builds a buffer from all data that is available
        according to the sampling mode specified. Default method just
        returns the whole registry
        """

        # if n_fix is a probability, then get an expected value of the number of views
        expected_views = n_fix if n_fix >= 1 else self.expected_n(n_fix)

        if self.objects_in_current_set == 'all':
            object_order = np.arange(self.n_objects)
        else:
            object_order = np.array(self.objects_in_current_set)

        if shuffle_object_order:
            np.random.shuffle(object_order)

        if sampling_mode == 'window':
            streambits = []
            for _ in range(approx_size // (round(expected_views) * self.n_objects)):
                for o in object_order:
                    # get the n_fix for each object
                    n_views = self.get_n(n_fix)
                    chosen_index = np.random.choice(
                        np.arange(0, self.n_views_per_object - n_views))
                    streambits.append(registry[registry.object_nr == o][
                        registry.time_idx.between(chosen_index, chosen_index + n_views - 1)])
                if shuffle_object_order:
                    np.random.shuffle(object_order)

            timestream = pd.concat(streambits, ignore_index=True)
            timestream.time_idx = np.arange(len(timestream.time_idx))

        elif sampling_mode == 'uniform':
            streambits = []
            for _ in range(approx_size // (round(expected_views) * self.n_objects)):
                for o in object_order:
                    # get the n_fix for each object
                    n_views = self.get_n(n_fix)
                    streambits.append(
                        registry.iloc[self.get_N_uniform_steps(n_views, o)])
                if shuffle_object_order:
                    np.random.shuffle(object_order)

            timestream = pd.concat(streambits, ignore_index=True)
            timestream.time_idx = np.arange(len(timestream.time_idx))

        elif sampling_mode == 'randomwalk':
            streambits = []
            for _ in range(approx_size // (round(expected_views) * self.n_objects)):
                for o in object_order:
                    # get the n_fix for each object
                    n_views = self.get_n(n_fix)
                    streambits.append(
                        registry.iloc[self.get_N_randomwalk_steps(n_views, o)])
                if shuffle_object_order:
                    np.random.shuffle(object_order)

            timestream = pd.concat(streambits, ignore_index=True)
            timestream.time_idx = np.arange(len(timestream.time_idx))

        elif sampling_mode == 'videowalk':
            streambits = []
            for _ in range(approx_size // (round(expected_views) * self.n_objects)):
                for o in object_order:
                    # get the n_fix for each object
                    n_views = self.get_n(n_fix)
                    streambits.append(
                        registry.iloc[self.get_N_videowalk_steps(n_views, o)])
                if shuffle_object_order:
                    np.random.shuffle(object_order)

            timestream = pd.concat(streambits, ignore_index=True)
            timestream.time_idx = np.arange(len(timestream.time_idx))

        else:
            print("[INFO] Warning, no sampling mode specified, defaulting to \
                whole dataset")
            timestream = registry  # if no mode, then return the whole registry

        return timestream

    def get_N_uniform_steps(self, N, object_nr):
        """
        Get index values of N uniform samples of a object specified by "object_nr".
        """
        chosen_indexs = np.random.choice(
            np.arange(0, self.n_views_per_object), N)
        return self.registry[self.registry.object_nr == object_nr].iloc[chosen_indexs].index

    def refresh_buffer(self):
        """
        refresh buffer takes an CLTTDataset class and refreshes its own buffer
        given the registry
        """
        self.buffer = self.build_buffer(
            self.registry, self.sampling_mode, self.n_fix, self.shuffle_object_order, self.buffer_size)
        pass

    def get_N_randomwalk_steps(self, N, object_nr):
        """
        Get index values of N random walk steps of a object specified by "object_nr".
        """
        raise Exception("Calling abstract method, please inherit \
        from the CLTTDataset class and reimplement this method")  # pseudoclass
        pass

    def get_N_videowalk_steps(self, N, object_nr):
        """
        Get index values of N random walk steps of a object specified by "object_nr".
        """
        raise Exception("Calling abstract method, please inherit \
        from the CLTTDataset class and reimplement this method")  # pseudoclass
        pass

    def expected_n(self, probability):
        """
        expected_n takes a float probability between 0 and 1
        and returns the expected value of the number of fixations
        """
        result = (1-probability)*(probability)/(1-(probability))**2 + 1
        return result

    def get_n(self, input):
        """
        get_n takes a float probability input between 0 and 1
        and returns n fixations according to probability
        if input >= 1 it just returns its argument
        """
        if input >= 1:
            return int(input)
        else:
            result = 1  # make sure that you switch to the next object once
            while input > np.random.random():
                result += 1
            return result


# datasets (CLTTDataset subclasses)
# -----

class CORE50Dataset(CLTTDataset):
    """
    CORE50Dataset adapts the CORE50 dataset (Lomonaco and Maltoni, 2017)
    and applies dynamic resampling to the video frames for unsupervised
    time-contrastive learning
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def get_dataset_properties(self):

        # basic properties (need to be there)
        self.n_objects = 50  # number of different objects >= n_classes
        self.n_classes = 10  # number of different classes
        self.labels = [
            "plug adapters",
            "mobile phones",
            "scissors",
            "light bulbs",
            "cans",
            "glasses",
            "balls",
            "markers",
            "cups",
            "remote controls",
        ]

        self.n_views_per_object = 2390  # how many overall views of each object
        # this is not exact and should not be used

        self.subdirectory = '/core50_128x128/'  # where is the dataset
        self.name = 'CORe50 Dataset'  # name of the dataset

        # custom properties (optional, dataset specific)
        # (anything you would want to have available in self)
        self.n_sessions = 11
        # directly get the parameter from config (TODO: Change this)
        self._dataset_percentage = config.TRAINING_PERCENTAGE

        # python properties which require getter and setter methods
        # directly get the parameter from config (TODO: Change this)
        self._n_fix_per_session = config.N_fix_per_session
        self._sample_across_sessions = False  # was False
        self._label_by = 'class'
        pass

    @property
    def dataset_percentage(self):
        return self._dataset_percentage

    @dataset_percentage.setter
    def dataset_percentage(self, p):
        old_percentage = self._dataset_percentage
        self._dataset_percentage = p
        # rebuild the registry
        self.registry = self.build_registry(self.split)
        # adapt buffer size to new dataset size
        self.buffer_size = len(self.registry)

        # rebuild the buffer
        if self.contrastive:
            self.refresh_buffer()
        else:
            # if used in non-contrastive mode the sampler just samples from all data
            self.buffer = self.registry
        pass

    @property
    def n_fix_per_session(self):
        return self._n_fix_per_session

    @n_fix_per_session.setter
    def n_fix_per_session(self, n):
        self._n_fix_per_session = n

        # rebuild the buffer
        if self.contrastive:
            self.refresh_buffer()
        else:
            # if used in non-contrastive mode the sampler just samples from all data
            self.buffer = self.registry
        pass

    @property
    def sample_across_sessions(self):
        return self._sample_across_sessions

    @sample_across_sessions.setter
    def sample_across_sessions(self, b):
        assert b in [True, False]
        self._sample_across_sessions = b

        # rebuild the buffer
        if self.contrastive:
            self.refresh_buffer()
        else:
            # if used in non-contrastive mode the sampler just samples from all data
            self.buffer = self.registry
        pass

    @property
    def label_by(self):
        return self._label_by

    @label_by.setter
    def label_by(self, l):
        assert l in ['class', 'object', 'session']
        self._label_by = l
        # change internal parameters of the dataset
        self.registry.drop('label', axis=1, inplace=True)
        if l == 'object':
            self.n_objects = 50
            self.n_classes = 50
            self.labels = [f'{i}' for i in range(50)]
            self.registry['label'] = self.registry['object_nr']
        elif l == 'session':
            # sessions are called 1-11, labels should be 0 - 10
            self.n_objects = self.n_sessions
            self.n_classes = self.n_sessions
            self.labels = [f'{i}' for i in range(1, self.n_sessions + 1)]
            self.registry['label'] = self.registry['session'] - 1
        else:
            self.get_dataset_properties()
            self.registry['label'] = self.registry['class']

        # rebuild the buffer
        if self.contrastive:
            self.refresh_buffer()
        else:
            # if used in non-contrastive mode the sampler just samples from all data
            self.buffer = self.registry
        pass

    def get_N_uniform_steps(self, N, object_nr):
        """
        Reimplementation of get_N_uniform steps to specify whether samples should
        be only of one session or across sessions
        """
        min_view = 0
        max_view = len(self.registry[self.registry.object_nr == object_nr]) - 1
        current_view = np.random.randint(0, max_view)

        current_session = self.registry[self.registry.object_nr ==
                                        object_nr].iloc[current_view].session
        max_view = len(self.registry[self.registry.object_nr ==
                                     object_nr][self.registry.session == current_session]) - 1

        if self._n_fix_per_session > N and self._n_fix_per_session >= 1.0:
            # do regular (fast) uniform sampling
            chosen_indexs = np.random.choice(np.arange(min_view, max_view), N)

            index = self.registry[self.registry.object_nr ==
                                  object_nr][self.registry.session == current_session].iloc[chosen_indexs].index

            return index
        else:
            # do complex uniform sampling in combination with cross-sessions
            counter = 0
            index = []
            while (counter < N):
                N_per_session = 0
                # count how many frames of one session you want to sample
                if (self._n_fix_per_session < 1.0):
                    while np.random.rand() < self._n_fix_per_session:
                        counter += 1
                        N_per_session += 1
                else:
                    if N - ((counter + 1)*int(self._n_fix_per_session)) >= 0:
                        counter += 1
                        N_per_session = int(self._n_fix_per_session)
                    else:
                        N_per_session = (
                            (counter + 1)*int(self._n_fix_per_session)) - N
                        counter = N

                # get N_per_session_views from the current session
                chosen_indexs = np.random.choice(
                    np.arange(min_view, max_view), N_per_session)
                index.append(self.registry[self.registry.object_nr == object_nr]
                             [self.registry.session == current_session].iloc[chosen_indexs].index)
                # new session that is not the old session
                other_sessions = set(self.registry.session.unique())
                other_sessions.remove(current_session)
                current_session = np.random.choice(list(other_sessions))
                max_view = len(self.registry[self.registry.object_nr ==
                                             object_nr][self.registry.session == current_session]) - 1

            index = np.concatenate(index)[:N]
            return index

    def get_N_randomwalk_steps(self, N, object_nr):
        """
        Get index values of N random walk steps of an object specified by "object_nr".
        """

        def walk(current_view, min_view, max_view):
            while True:
                # 2 possible direction in which to go from the current position
                # Possible steps: +: left, -: right
                rand = np.random.randint(low=0, high=2)
                if (rand == 0) & (current_view > min_view):
                    current_view -= 1
                    break
                if (rand == 1) & (current_view < max_view):
                    current_view += 1
                    break
            return current_view

        position = []
        index = []
        # get the object number and get a random index from there since not every object has the same amount of views
        min_view = 0
        max_view = len(self.registry[self.registry.object_nr == object_nr]) - 1
        current_view = np.random.randint(min_view, max_view)
        current_session = self.registry[self.registry.object_nr ==
                                        object_nr].iloc[current_view].session

        max_view = len(self.registry[self.registry.object_nr ==
                                     object_nr][self.registry.session == current_session]) - 1
        current_view = np.random.randint(min_view, max_view)

        counter = 0

        while (counter < N):
            if (self._n_fix_per_session < 1.0):
                while np.random.rand() < self._n_fix_per_session:
                    current_view = walk(current_view, min_view, max_view)
                    position.append(current_view)
                    counter += 1
            else:
                for _ in range(self._n_fix_per_session):
                    current_view = walk(current_view, min_view, max_view)
                    position.append(current_view)
                    counter += 1

            # add first view by default if chosen to switch session
            if len(position) == 0:
                position.append(current_view)
                counter += 1
            index.append(self.registry[self.registry.object_nr == object_nr]
                         [self.registry.session == current_session].iloc[position].index)
            # new session that is not the old session
            other_sessions = set(self.registry.session.unique())
            other_sessions.remove(current_session)
            current_session = np.random.choice(list(other_sessions))
            max_view = len(self.registry[self.registry.object_nr ==
                                         object_nr][self.registry.session == current_session]) - 1
            current_view = np.random.randint(min_view, max_view)
            position = []
        index = np.concatenate(index)[:N]
        return index

    def get_N_videowalk_steps(self, N, object_nr):
        """
        Get index values of N random walk steps of an object specified by "object_nr".
        """

        def walk(current_view, min_view, max_view, direction):
            while True:
                if (direction == 1) & (current_view < max_view):
                    current_view += 1
                    break
                elif (direction == -1) & (current_view > min_view):
                    current_view -= 1
                    break
                else:
                    direction *= (-1)
            return current_view, direction

        position = []
        index = []
        # get the object number and get a random index from there since not every object has the same amount of views
        min_view = 0
        max_view = len(self.registry[self.registry.object_nr == object_nr]) - 1
        current_view = np.random.randint(min_view, max_view)
        current_session = self.registry[self.registry.object_nr ==
                                        object_nr].iloc[current_view].session

        max_view = len(self.registry[self.registry.object_nr ==
                                     object_nr][self.registry.session == current_session]) - 1
        current_view = np.random.randint(min_view, max_view)

        counter = 0
        direction = 1 if np.random.random() > 0.5 else -1

        while (counter < N):
            if (self._n_fix_per_session < 1.0):
                while np.random.rand() < self._n_fix_per_session:
                    current_view, direction = walk(
                        current_view, min_view, max_view, direction)
                    position.append(current_view)
                    counter += 1
            else:
                for _ in range(self._n_fix_per_session):
                    current_view, direction = walk(
                        current_view, min_view, max_view, direction)
                    position.append(current_view)
                    counter += 1

            # add first view by default if chosen to switch session
            if len(position) == 0:
                position.append(current_view)
                counter += 1
            index.append(self.registry[self.registry.object_nr == object_nr]
                         [self.registry.session == current_session].iloc[position].index)
            # new session that is not the old session
            other_sessions = set(self.registry.session.unique())
            other_sessions.remove(current_session)
            current_session = np.random.choice(list(other_sessions))
            max_view = len(self.registry[self.registry.object_nr ==
                                         object_nr][self.registry.session == current_session]) - 1
            current_view = np.random.randint(min_view, max_view)
            position = []
        index = np.concatenate(index)[:N]
        return index

    def build_registry(self, split):
        """
        Reimplementation of the build_registry method, because CORE50
        is structured in sessions of objects and does not have a fixed
        number of views per objects
        """

        def label_from_object(object_number):
            """
            label_from_object is a helper function that takes the the int
            object_number and returns the int label_number
            """
            return (object_number) // 5

        path_list = []
        object_list = []
        label_list = []
        time_list = []
        session_list = []
        episode_list = []  # an episode is one timestream of one object
        e = 0

        # by default training and testing is split into
        # sessions: training: 1,2,4,5,6,8,9,11; test: 3,7,10
        # to get additional insight into learning progress
        # an additional validation set is generated from 10%
        # of the training sessions, every 10 image gets
        # sorted towards validation

        if split == 'train':
            sessions = [1, 2, 4, 5, 6, 8, 9, 11]
            objects = range(self.n_objects)
            def sample_in_split(id): return not(id % 10 == 0)
        elif split == 'test':
            sessions = [3, 7, 10]
            objects = range(self.n_objects)
            def sample_in_split(id): return True
        elif split == 'val':
            sessions = [1, 2, 4, 5, 6, 8, 9, 11]
            objects = range(self.n_objects)
            def sample_in_split(id): return (id % 10 == 0)
        elif 'train_alt' in split:
            k = int(split.split('_')[-1])
            sessions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            objects = [o for o in range(self.n_objects) if (o-k) % 5]

            def sample_in_split(id): return not(id % 10 == 0)
        elif 'test_alt' in split:
            k = int(split.split('_')[-1])
            sessions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            def sample_in_split(id): return True
            objects = [o for o in range(self.n_objects) if not((o-k) % 5)]
        elif 'val_alt' in split:
            k = int(split.split('_')[-1])
            sessions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            objects = [o for o in range(self.n_objects) if (o-k) % 5]

            def sample_in_split(id): return (id % 10 == 0)
        else:
            raise ValueError(
                f'[INFO:] Chosen split {split} is not defined in this dataset')

        self.objects_in_current_set = objects

        for s in sessions:
            for o in objects:
                d = '/s{}/o{}/'.format(s, o+1)
                d = self.root + self.subdirectory + d
                # have an ordered list
                list_of_files = os.listdir(d)
                list_of_files.sort()

                # filter for images using png suffix
                list_of_files = [
                    f for f in list_of_files if f.endswith('.png')]

                # only take a split of each session in order to
                # have a reduced dataset
                list_of_files = list_of_files[:int(
                    len(list_of_files) * self._dataset_percentage)]

                for path in list_of_files:
                    full_path = os.path.join(d, path)
                    if os.path.isfile(full_path):
                        current_time_id = int(
                            path.split('.')[0].rsplit('_')[-1])
                        if sample_in_split(current_time_id):
                            path_list.append(full_path)
                            object_list.append(o)
                            session_list.append(s)
                            time_list.append(current_time_id)
                            label_list.append(
                                label_from_object(o))
                            episode_list.append(e)
                e += 1

        tempdict = {'path_to_file': path_list, 'class': label_list, 'object_nr': object_list,
                    'session': session_list, 'time_idx': time_list, 'episode': episode_list, 'label': label_list, }

        dataframe = pd.DataFrame(tempdict)
        # prepare for cross-session sampling

        dataframe['indices'] = dataframe.index
        return dataframe


# ----------------
# main program
# ----------------
if __name__ == "__main__":

    # CORE50 Dataset
    # -----

    dataset = CORE50Dataset(
        root='../data',
        split='train_alt_0',
        transform=transforms.ToTensor(),
        n_fix=10,
        contrastive=True,
        sampling_mode='randomwalk',
    )

    # original timeseries
    dataloader = DataLoader(dataset, batch_size=100,
                            num_workers=0, shuffle=False)
    for ibatch, sample_batched in enumerate(dataloader):
        show_batch(sample_batched)
        if ibatch == 2:
            break

    # shuffled timeseries
    dataloader = DataLoader(dataset, batch_size=100,
                            num_workers=0, shuffle=True)
    for ibatch, sample_batched in enumerate(dataloader):
        show_batch(sample_batched)
        if ibatch == 2:
            break


# _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
