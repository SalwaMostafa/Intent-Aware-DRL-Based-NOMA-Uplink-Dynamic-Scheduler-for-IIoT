import gym.spaces as spaces
import enum
import numpy as np


def to_categorical(y, num_classes=None, dtype="float32"):    
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def to_almost_categorical(y, num_classes=None, dtype="float32"):
    y = np.array(y, dtype="int")
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.ones((n, num_classes), dtype=dtype) * (0.2 / (num_classes - 1))
    categorical[np.arange(n), y] = 0.8
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def get_len_space(action_space):
    """Get the total dimension of a flattened action in the action_space.
    Parameters
    ----------
    action_space : gym.spaces.Space
        Action space
    Returns
    -------
    int
        Number of elements in the flattened array of an action.
    """
    # MultiDiscrete is just the sum of elements:
    if isinstance(action_space, spaces.MultiDiscrete):
        return sum(action_space.nvec)
    # MultiBinary has two cases, with one element in the input and more:
    elif isinstance(action_space, spaces.MultiBinary):
        if isinstance(action_space.n, list):
            return np.prod(action_space.n)
        else:
            return action_space.n
    # Discrete is just the number of options:
    elif isinstance(action_space, spaces.Discrete):
        return action_space.n
    # Tuple is a recursive solution:
    elif isinstance(action_space, spaces.Tuple):        
        return sum([get_len_space(aa) for aa in action_space])


def relaxed_to_one_hot(in_list):
    out_list = []
    for vector in in_list:
        ar = np.zeros(vector.shape)
        ar[np.argmax(vector)] = 1
        out_list.append(ar)
    return out_list


class EnvInputTransformer:
    def __init__(self,env,ea_hist_size=0,comm_hist_size=0,ca_hist_size=0,obs_hist_size=0,ul_voc_size=2,dl_voc_size=3,):
        
        self.env = env
        self.comm_hist_size = comm_hist_size
        self.obs_hist_size = obs_hist_size
        self.ca_hist_size = ca_hist_size
        self.ea_hist_size = ea_hist_size
        self.ul_voc_size = ul_voc_size
        self.dl_voc_size = dl_voc_size
        n_ues = env.n_ues
        n_actions = env.nA
        # Base station state is a tuple (obs, obs_n, uci, dci),
        # each field is one-hot-encoded
        state_bs_dim = (obs_hist_size + 1) * (n_ues + 2)
        state_bs_dim += ul_voc_size * n_ues * comm_hist_size
        state_bs_dim += dl_voc_size * n_ues * ca_hist_size
        s_bs = spaces.MultiBinary(state_bs_dim)
        # Observation space of each user is the status of the buffer:
        state_ue_dim = [env.tx_buffer_capacity] * (obs_hist_size + 1)
        # The UE agent space also includes the previous DL msgs received (Categorical):
        state_ue_dim.extend([2] * dl_voc_size * comm_hist_size)
        # The agent state includes the previous env act. taken by the UE (Categorical):
        state_ue_dim.extend([2] * n_actions * ea_hist_size)
        # The agent state includes the previous com actions taken by the UE:
        state_ue_dim.extend([2] * voc_size_ul * ca_hist_size)
        # State Space:
        s_ues = (spaces.MultiDiscrete(state_ue_dim) for ue in range(n_ues))
        # Joint state space (bs, ue_1, ..., ue_n):
        self.observation_space = spaces.Tuple((s_bs, *s_ues))

    def reset(self):
        obs = self.env.reset()
        n_ues = self.env.n_ues
        # Get greater size:
        greater_size = max(self.ca_hist_size, self.comm_hist_size)
        # Initialize Buffers:
        ac_buffer = deque([[0] * n_ues] * self.ea_hist_size, maxlen=self.ea_hist_size)
        ul_msg_buffer = deque([[0] * n_ues] * self.ca_hist_size, maxlen=greater_size)
        dl_msg_buffer = deque([[0] * n_ues] * self.ca_hist_size, maxlen=greater_size)
        obs_buffer = deque([0] * (n_ues + 1), maxlen=self.obs_hist_size)
        agents_state = self.transform()

    def step(self, actions):
        actions = 1


class Side(enum.IntEnum):
    NONE = 0
    LEFT = 1
    RIGHT = 2


class ArrayBuffer (object):

    def __init__(self, dtype=np.float32, init_shape=(2048,)):

        if isinstance(init_shape, int):
            init_shape = (init_shape,)
        self._dtype = dtype
        self._init_shape = init_shape
        self._array = np.zeros(init_shape, dtype)
        self._start = 0
        self._stop = 0
        self._extensions = np.zeros(2048, dtype=[('left', np.int32),('right', np.int32),('lenght', np.int32)])
        self._extension_n = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def nbytes(self):
        return self._array.nbytes

    @property
    def itemsize(self):
        return self._array.itemsize

    def __len__(self):
        return self._stop - self._start

    def __getitem__(self, key):
        return self._array[self._start:self._stop][key]

    def __setitem__(self, key, value):
        self._array[self._start:self._stop][key] = value

    def _check_enlarge(self, side, extlen):

        if side:
            self._extensions[self._extension_n] = (extlen if side == Side.LEFT else 0,extlen if side == Side.RIGHT else 0,
                                                   len(self) + extlen)
            
            self._extension_n = ((self._extension_n + 1) %
                                 (len(self._extensions)))

        if side == Side.RIGHT and len(self._array) - self._stop >= extlen:
            return
        elif side == Side.LEFT and self._start >= extlen:
            return

        left_ext = np.sum(self._extensions['left'])
        right_ext = np.sum(self._extensions['right'])
        max_lenght = np.max(self._extensions['lenght'])

        mult = np.int32(np.ceil(max_lenght / max(1, left_ext + right_ext)))

        left_ext *= mult
        right_ext *= mult

        newshape = (left_ext + max(len(self), self._init_shape[0]) + extlen + right_ext,*self._array.shape[1:])

        if side == Side.LEFT:
            newstop = newshape[0] - right_ext
            newstart = newstop - len(self)
        else:
            newstart = left_ext
            newstop = newstart + len(self)

        assert newstop - newstart == len(self)

        newarr = np.zeros(newshape, self._dtype)
        if newstart < newstop:
            newarr[newstart:newstop] = self[:]
        self._start = newstart
        self._stop = newstop
        self._array = newarr

    def extend(self, other):
        ol = len(other)
        self._check_enlarge(Side.RIGHT, ol)
        self._array[self._stop:self._stop + ol] = other[:]
        self._stop += ol

    def append(self, other):
        ol = 1
        self._check_enlarge(Side.RIGHT, ol)
        self._array[self._stop:self._stop + ol] = other
        self._stop += ol


    def extendleft(self, other):
        ol = len(other)
        self._check_enlarge(Side.LEFT, ol)
        self._array[self._start - ol:self._start] = other[:]
        self._start -= ol
    
    def appendleft(self, other):
        ol = 1
        self._check_enlarge(Side.LEFT, ol)
        self._array[self._start - ol:self._start] = other
        self._start -= ol

    def pop(self, count):
        count = min(count, len(self))
        ret = self[-count:]
        self._stop -= count
        return ret

    def popleft(self, count):
        count = min(count, len(self))
        ret = self[:count]
        self._start += count
        return ret

    def clear(self):
        self._start = 0
        self._stop = 0
        self._check_enlarge(Side.NONE, 0)
