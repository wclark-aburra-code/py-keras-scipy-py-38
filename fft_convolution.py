#from singledispatch import singledispatch # type: ignore
import wave, numpy              # type: ignore
from scipy import signal        # type: ignore
import utility                  # type: ignore
import wavefile                 # type: ignore
from scipy.io import wavfile    # type: ignore 
from scipy import fftpack       # type: ignore
import math
from typing import Union, List, runtime_checkable, Protocol, Sized
from functools import reduce
#from pampy import match, REST, _ # type: ignore
from multimethod import multimethod  # type: ignore
from abc import abstractmethod
SignalVector = Union[List[float], List[int], List[complex], numpy.ndarray]

Output = Union[List[SignalVector], SignalVector]

Numeric = Union[int, float]


def unit_impulse() -> numpy.array:
    return numpy.array([1.])

# Utility functions

def nextPowerOf2(num: int, i:int = 1) -> int:
    while i < num:
        i *= 2
    return i

def padded(byte_array: Union[numpy.ndarray, List[Numeric]], output_length: int) -> numpy.ndarray:
    return numpy.append(byte_array,numpy.zeros(output_length-len(byte_array))) # check - append, not concat

def hanning_multiplier(i: int, block_size: int) -> float:
    return 0.5 * (1. - numpy.cos(2.0*numpy.pi*i/(block_size-1)))


def complex_coefficients(signal: SignalVector) -> List[float]:
    return [abs(cpx) for cpx in fftpack.fft(signal)]


class Signal(Protocol):
    byte_content: Union[numpy.ndarray, List[numpy.ndarray]]
    def __init__(self, byte_content: Union[numpy.ndarray, List[numpy.ndarray]]):
        self.byte_content = byte_content

#@runtime_checkable
class MonoSignal(Signal):
    byte_content: numpy.ndarray


# implement Sized as well?
class MultichanSignal(Signal):
    byte_content: List[numpy.ndarray]


# Structural types
@runtime_checkable 
class Filter(Protocol):
    @abstractmethod
    def raw_signal_channels(self):
        [[]]
# rm runtime_checkable annotation
@runtime_checkable # not necessary if this comes before?
class MonoElement(Protocol):
    @abstractmethod
    def as_convolved_filter(self) -> List[numpy.ndarray]:
        return []

 # not necessary if this comes before?
class MonoFilter(Filter,MonoElement):
    ir_length: int
    output_length: int
    window_size: int
    byte_array: numpy.ndarray
    complex_phasors: numpy.ndarray    
    def raw_signal_channels(self):
        return [byte_array]
    def as_convolved_filter(self) -> List[numpy.ndarray]:
        return []

class MonoFilter(Filter, MonoElement): # we want a monoprotocol, for as_convolved_filter
    ir_length: int
    output_length: int
    window_size: int
    byte_array: numpy.ndarray
    complex_phasors: numpy.ndarray
    def __init__(self, byte_array: SignalVector): #output length must be pwr of 2 (RAISE)
        self.ir_length = len(byte_array)
        self.output_length = nextPowerOf2(self.ir_length)
        self.window_size = self.output_length - self.ir_length + 1
        self.byte_array = padded(byte_array, self.output_length)
        self.complex_phasors = fftpack.fft(self.byte_array)
    def step_response(self) -> numpy.ndarray:
        step_impulse = numpy.ones(len(self.byte_array))
        return signal.fftconvolve(step_impulse, self.byte_array)
    def normalization_factor(self) -> float:
        return self.step_response().max()
    def as_convolved_filter(self):
        # RENAME
        return self.byte_array        
    def raw_signal_channels(self):
        return [self.byte_array]


class MultichanFilter(Filter, Sized):
    filters: List[MonoFilter]
    quantity: int
    def __init__(self, filter_array: List[MonoFilter]):
        self.filters = filter_array
        self.quantity = len(filter_array)
    def __len__(self):
        return self.quantity
    def raw_signal_channels(self):
        return [filter.byte_array for filter in self.filters]


class MonoEffectChain(Filter, MonoElement):
    filter_list: List[MonoFilter]
    def __init__(self, filter_list: List[MonoFilter]):
        self.filter_list = filter_list
    def as_convolved_filter(self) -> numpy.ndarray:
        return reduce((lambda sig1, sig2: signal.fftconvolve(sig1, sig2)), [unit_impulse()] + [f.byte_array for f in self.filter_list] )
    def raw_signal_channels(self):
        return [self.as_convolved_filter()]

class MultiEffectChain(Filter, Sized):
    channels: List[MonoEffectChain]
    quantity: int
    def __init__(self, channels: List[MonoEffectChain]):
        self.channels = channels
        self.quantity = len(self.channels)
    def __len__(self):
        return self.quantity      
    def raw_signal_channels(self):
        return [channel.as_convolved_filter() for channel in self.channels]
        # TRY return MultichanSignal([channel.as_convolved_filter() for channel in self.channels])

FilterOutput = Union[numpy.ndarray, List[numpy.ndarray]]

# what about this case -- [numpy.ndarray]... like a "literal multichannel signal"
#???
@multimethod
def convolve(signal1, signal2) -> FilterOutput:
    return numpy.array([])


# cant do this! can't use List for matching
#@convolve.register
#def _alias16(signal_array: List[numpy.ndarray], filter: Filter) -> List[numpy.ndarray]:
#    return convolve(MultichanFilter([MonoFilter(chan) for sig in signal_array]), filter)

# TRY WRAPPING SINGLE CASE UNION TYPE SIGNAL AGAIN..
# WE HAD A BUG BELOW WITH REUSING "SIGNAL" NAME OF LIBRARY
# FROM SCIPY
#@convolve.register
#def _alias16(signal: MonoSignal, filter: Filter) -> numpy.ndarray:
#    return convolve(signal.byte_content, filter) 

#@convolve.register # this is a linguistic sort of hack -- a signal and a filter are the same datatype essentially, and each can be convolved as if it were the other
#def _alias17(signal: MultichanSignal, filter: Filter) -> List[numpy.ndarray]:
#    return convolve(MultichanFilter([MonoFilter(chan) for chan in signal.byte_content]), filter)     

#@convolve.register
#def _alias20(filter: Filter, signal1: MonoSignal) -> numpy.ndarray:
#    return convolve(filter, signal1.byte_content) 

#@convolve.register # this is a linguistic sort of hack -- a signal and a filter are the same datatype essentially, and each can be convolved as if it were the other
#def _alias21(filter: Filter, signal1: MultichanSignal) -> List[numpy.ndarray]:
#    return convolve(filter, MultichanFilter([MonoFilter(chan) for chan in signal.byte_content]))     

#@convolve.register
#def _alias22(signal1: MonoSignal, sig2: Signal) -> Signal:
#    return convolve(signal.byte_content, sig2) 

#@convolve.register # this is a linguistic sort of hack -- a signal and a filter are the same datatype essentially, and each can be convolved as if it were the other
#def _alias23(signal1: MultichanSignal, sig2: Signal) -> List[numpy.ndarray]:
#    return convolve(MultichanFilter([MonoFilter(chan) for chan in signal.byte_content]), sig2)     


#@convolve.register
#def _alias24(signal1: MonoSignal, sig2: numpy.ndarray) -> Signal:
#    return convolve(signal1, MonoFilter(sig2)) 

#@convolve.register # this is a linguistic sort of hack -- a signal and a filter are the same datatype essentially, and each can be convolved as if it were the other
#def _alias25(signal1: MultichanSignal, sig2: numpy.ndarray) -> Signal:
#    return convolve(signal1, MonoFilter(sig2))

# can't convolve literals
# can't convolve multielement with multieffect

multitap_delay = numpy.array([1., 0., 0.6, 0., 0.4, 0., 0.2])
slapback_delay = numpy.array([1., 0.8])
basic_reverb = numpy.array([0., .1, -.1, .2, -.1, .2, -.1])
turn_up = numpy.array([2.])
lowpass_filter = numpy.array([1., 0.2, 0.2, 0.2, 0.2, 0.2]) # moving average
linear_phase_highpass_filter = numpy.array([-0.2, -0.2, 1., -0.2, -0.2]) # incidentally, this is a valid highpass filter for any signal. Try it on some audio, using wavesample.
mute = numpy.array([0.])
hpf_differential = numpy.array([1., -.1]) # like a high pass filter

lpf_wrapped = MonoFilter(lowpass_filter)
stereo_lpf = MultichanFilter([lpf_wrapped, lpf_wrapped])
reverb_wrapped = MonoFilter(basic_reverb)
muffled_reverb = MonoEffectChain([lpf_wrapped, reverb_wrapped])
stereo_muffled_reverb = MultiEffectChain([muffled_reverb, muffled_reverb])


# basically working... we think... dispatch seems working at least


# MUST PAD MORE... FAILS FOR VERY LOW LIST LENGTHS
def overlap_add(filter_object: MonoFilter, dry_signal: SignalVector):
    num_sections = math.floor(len(dry_signal)/filter_object.window_size)
    if num_sections * filter_object.window_size != len(dry_signal) :
        dry_signal = padded(dry_signal,(num_sections+1)*filter_object.window_size)
        num_sections += 1
    overlap_length = filter_object.ir_length - 1
    sections_list = numpy.split(dry_signal, num_sections)
    overlap_kernel = numpy.zeros(overlap_length)
    output = numpy.array([])
    for section in sections_list:
        section_length = len(section)
        padded_section = numpy.zeros(filter_object.output_length)
        for n, sample in enumerate(section):
            padded_section[n] = hanning_multiplier(n,section_length)*section[n]
        section_phasors = fftpack.fft(padded_section)
        convolved_section_phasors = numpy.multiply(section_phasors, filter_object.complex_phasors) # dbl check complex multiplication
        convolved_section_samples = fftpack.ifft(convolved_section_phasors)
        overlapped_sum = numpy.copy(convolved_section_samples[0:section_length]) # must copy, we refer to this later
        for n, _ in enumerate(overlapped_sum):
            overlapped_sum[n] += overlap_kernel[n]
        output = numpy.append(output, overlapped_sum)
        overlap_kernel = convolved_section_samples[section_length:overlap_length]
    return numpy.real(output)

class OverlapAddingMachine:
    def __init__(self, filter_object: MonoFilter, dry_signal: SignalVector):
        self.filter_object = filter_object
        self.num_sections = math.floor(len(dry_signal)/filter_object.window_size)
        self.dry_signal = dry_signal
        if self.num_sections * filter_object.window_size != len(dry_signal) :
            self.dry_signal = padded(self.dry_signal,(self.num_sections+1)*self.filter_object.window_size)
            self.num_sections += 1
        self.overlap_length = filter_object.ir_length - 1
        self.sections_list = numpy.split(self.dry_signal, self.num_sections)
        self.overlap_kernel = numpy.zeros(self.overlap_length)
        self.output = numpy.array([])
        self.section_num = 0
        self.section_length = len(self.sections_list[0])    
    def __next__(self):        
        if (self.section_num+1 < self.num_sections):
            self.section_num += 1
            section = self.sections_list[self.section_num]
            padded_section = numpy.zeros(self.filter_object.output_length)
            for n, sample in enumerate(section):
                padded_section[n] = hanning_multiplier(n,self.section_length)*section[n]
            section_phasors = fftpack.fft(padded_section)
            convolved_section_phasors = numpy.multiply(section_phasors, self.filter_object.complex_phasors)
            convolved_section_samples = fftpack.ifft(convolved_section_phasors)
            overlapped_sum = numpy.copy(convolved_section_samples[0:self.section_length])
            for n, _ in enumerate(overlapped_sum):
                overlapped_sum[n] += self.overlap_kernel[n]
            self.overlap_kernel = convolved_section_samples[self.section_length:self.overlap_length]
            return numpy.real(overlapped_sum)
        else:
            raise StopIteration


# sine generator for tests
def sin_signal(freq: Numeric, length: int):
    x = numpy.linspace(0,length-1) # or arange
    return numpy.sin(2*(numpy.pi)*freq*x)

# naive IR's for tests -- ADD *LINKS* TO EXPLAIN
multitap_delay = numpy.array([1., 0., 0.6, 0., 0.4, 0., 0.2])
slapback_delay = numpy.array([1., 0.8])
basic_reverb = numpy.array([0., .1, -.1, .2, -.1, .2, -.1])
turn_up = numpy.array([2.])
lowpass_filter = numpy.array([1., 0.2, 0.2, 0.2, 0.2, 0.2]) # moving average
linear_phase_highpass_filter = numpy.array([-0.2, -0.2, 1., -0.2, -0.2]) # incidentally, this is a valid highpass filter for any signal. Try it on some audio, using wavesample.
mute = numpy.array([0.])
hpf_differential = numpy.array([1., -.1]) # like a high pass filter

# *LINKS* TO EXPLAIN
def input_side_convolve(sig1: SignalVector, sig2: SignalVector):
    output = numpy.zeros(len(sig1) + len(sig2) - 1)
    for num1, sample1 in enumerate(sig1):
        for num2, sample2 in enumerate(sig2):
            output[num1 + num2] += sample1 * sample2
    return output

class WaveSample:        
    def __init__(self, filename: str):
        wf = wavefile.load(filename=filename)
        self.signal_vector = wf[1]
        if self.signal_vector.shape[0] == 2:
            self.left = self.signal_vector[0]
            self.right = self.signal_vector[1]            
        elif self.signal_vector.shape[0] == 1:
            self.mono = self.signal_vector[0]

# local manual tests USE WAVESAMPLE CLASS!
def test_local_files():
    input_signal = wavefile.load(filename="./audio_dataset/test/hi_hat/ALCHH36.WAV")
    impulse_response = wavefile.load(filename="./impulse_responses/spaceEchoIR.wav")  # already floating point bytearray
    second_IR = wavefile.load(filename="./impulse_responses/echo2IR.wav")
    (left, right) = input_signal[1]
    output_signal = signal.fftconvolve(input_signal[1][0], impulse_response[1][0]) 
    output_signal2 = signal.fftconvolve(output_signal, second_IR[1][0])
    wavfile.write("./audio_dataset/convolved_hihat.wav", 44100, utility.float2pcm(output_signal2))



# copied

# Structural types
@runtime_checkable 
class Filter(Protocol):
    @abstractmethod
    def raw_signal_channels(self):
        [[]]
# rm runtime_checkable annotation
@runtime_checkable # not necessary if this comes before?
class MonoElement(Protocol):
    @abstractmethod
    def as_convolved_filter(self) -> List[numpy.ndarray]:
        return []


class MonoFilter(Filter, MonoElement): # we want a monoprotocol, for as_convolved_filter
    ir_length: int
    output_length: int
    window_size: int
    byte_array: numpy.ndarray
    complex_phasors: numpy.ndarray
    def __init__(self, byte_array: SignalVector): #output length must be pwr of 2 (RAISE)
        self.ir_length = len(byte_array)
        self.output_length = nextPowerOf2(self.ir_length)
        self.window_size = self.output_length - self.ir_length + 1
        self.byte_array = padded(byte_array, self.output_length)
        self.complex_phasors = fftpack.fft(self.byte_array)
    def step_response(self) -> numpy.ndarray:
        step_impulse = numpy.ones(len(self.byte_array))
        return signal.fftconvolve(step_impulse, self.byte_array)
    def normalization_factor(self) -> float:
        return self.step_response().max()
    def as_convolved_filter(self):
        # RENAME
        return self.byte_array        
    def raw_signal_channels(self):
        return [self.byte_array]

@convolve.register
def _alias0(signal1: numpy.ndarray, signal2: numpy.ndarray) -> numpy.ndarray:
    return signal.fftconvolve(signal1, signal2)
@convolve.register
def _alias3(filter1: MonoFilter, filter: numpy.ndarray) -> numpy.ndarray:
    return  convolve(filter1.byte_array, filter)    

input_signal = wavefile.load(filename="./audio_dataset/test/hi_hat/ALCHH36.WAV")
impulse_response = wavefile.load(filename="./impulse_responses/spaceEchoIR.wav")  # already floating point bytearray
second_IR = wavefile.load(filename="./impulse_responses/echo2IR.wav")
(left) = input_signal[1]
wv = wavefile.load("/users/usuario/Desktop/bad.wav")
(trackleft, trackright) = wv[1]

(irleft) = impulse_response[1]
mf = MonoFilter(irleft)
#    convolve(trackleft, mf)
#    convolve(mf, trackleft)
[irrealleft] = irleft
convolve(trackleft, irrealleft)
convolve(MonoFilter(irrealleft), trackleft)
convolve(trackleft, MonoFilter(irrealleft))
mf = MonoFilter(irrealleft)
class MultichanFilter(Filter, Sized):
    filters: List[MonoFilter]
    quantity: int
    def __init__(self, filter_array: List[MonoFilter]):
        self.filters = filter_array
        self.quantity = len(filter_array)
    def __len__(self):
        return self.quantity
    def raw_signal_channels(self):
        return self.filters
 mcf = MultichanFilter([irrealleft,irrealleft])

@convolve.register
def _alias2(signal1: numpy.ndarray, filter: MultichanFilter) -> List[numpy.ndarray]:
    return [signal.fftconvolve(signal1, filterbytes) for filterbytes in filter.filters]
convolve(trackleft, mcf)


@convolve.register
def _alias99(filter1: MonoFilter, filter: MonoFilter) -> numpy.ndarray:
    return  signal.fftconvolve(filter1.byte_array, filter.byte_array)
convolve(mf,mf)


@convolve.register
def _alias4(mono_filter: MonoFilter, multi_filter: MultichanFilter) -> List[numpy.ndarray]:
    return  [convolve(mono_filter, chan_filter) for chan_filter in multi_filter.filters]
convolve(mf,mcf)
@convolve.register
def _alias5(multi_filter: MultichanFilter, mono_filter: MonoFilter) -> List[numpy.ndarray]:
    return  [convolve(chan_filter, mono_filter) for chan_filter in multi_filter.filters]
convolve(mcf,mf)

@convolve.register
def _alias6(multi_filter: MultichanFilter, multi_filter2: MultichanFilter) -> List[numpy.ndarray]:
    if len(multi_filter) != len(multi_filter2):
        raise Exception("channel number mismatch")
    filter1_channels = multi_filter.raw_signal_channels()
    filter2_channels = multi_filter2.raw_signal_channels()
    return [signal.fftconvolve(f1, f2) for (f1,f2) in list(zip(filter1_channels, filter2_channels))]

convolve(mcf, mcf)

class MonoEffectChain(Filter, MonoElement):
    filter_list: List[MonoFilter]
    def __init__(self, filter_list: List[MonoFilter]):
        self.filter_list = filter_list
    def as_convolved_filter(self) -> numpy.ndarray:
        return reduce((lambda sig1, sig2: signal.fftconvolve(sig1, sig2)), [unit_impulse()] + [f.byte_array for f in self.filter_list] )
    def raw_signal_channels(self):
        return [self.as_convolved_filter()]

class MultiEffectChain(Filter, Sized):
    channels: List[MonoEffectChain]
    quantity: int
    def __init__(self, channels: List[MonoEffectChain]):
        self.channels = channels
        self.quantity = len(self.channels)
    def __len__(self):
        return self.quantity      
    def raw_signal_channels(self):
        return [channel.as_convolved_filter() for channel in self.channels]
        # TRY return MultichanSignal([channel.as_convolved_filter() for channel in self.channels])
mec = MonoEffectChain([mf, mf, mf])

@convolve.register
def _alias7(signal1: numpy.ndarray, chain: MonoEffectChain) -> numpy.ndarray:
    return signal.fftconvolve(signal1, chain.as_convolved_filter())

@convolve.register
def _alias8(signal1: MonoFilter, chain: MonoEffectChain) -> numpy.ndarray:
    return signal.fftconvolve(signal1.byte_array, chain.as_convolved_filter())
convolve(mf, mec)
 convolve(mf.byte_array, mec)




@convolve.register
def _alias9(signal1: MonoFilter, chain: MultiEffectChain) -> List[numpy.ndarray]:
    return [convolve(signal1.as_convolved_filter(), chan) for chan in chain.raw_signal_channels()]

@convolve.register
def _alias10(signal1: numpy.ndarray, chain: MultiEffectChain) -> List[numpy.ndarray]:
    return convolve(MonoFilter(signal1), chain)

@convolve.register
def _alias11(monochain: MonoEffectChain, multichain: MultiEffectChain) -> List[numpy.ndarray]:
    return convolve(monochain.as_convolved_filter(), multichain)

@convolve.register
def _alias19(monochain: MonoEffectChain, monofilter: MonoFilter) -> numpy.ndarray:
    return convolve(monochain.as_convolved_filter(), monofilter)

@convolve.register
def _alias18(monochain: MonoEffectChain, raw_signal: numpy.ndarray) -> numpy.ndarray:
    return convolve(monochain.as_convolved_filter(), raw_signal)

@convolve.register
def _alias12(multichain: MultiEffectChain, monochain: MonoEffectChain) -> List[numpy.ndarray]:
    return convolve(multichain, monochain.as_convolved_filter())

@convolve.register
def _alias13(multichain: MultiEffectChain, multichain2: MultiEffectChain) -> List[numpy.ndarray]:
    if (len(multichain) != len(multichain2)):
        raise Exception("channel number mismatch")
    return [convolve(f1, f2) for (f1,f2) in list(zip(multichain.channels, multichain2.channels))]

@convolve.register
def _alias15(multichain: MultiEffectChain, monofilter: MonoFilter) -> List[numpy.ndarray]:
    return [convolve(ch, monofilter) for ch in multichain.channels ]

@convolve.register
def _alias14(multichain: MultiEffectChain, mono_signal: numpy.ndarray) -> List[numpy.ndarray]:
    return [convolve(ch, mono_signal) for ch in multichain.channels ]

@convolve.register
def _alias16(signal1: MultichanSignal, filter: Filter) -> List[numpy.ndarray]:
    return convolve(MultichanFilter([MonoFilter(sig) for sig in signal1.byte_content]), filter)

@convolve.register
def _alias17(signal1: MonoSignal, filter: Filter) -> List[numpy.ndarray]:
    return convolve(MonoFilter(signal1.byte_content), filter)
# seems to work... try wrapping output... anything that wasn't all ndarray gets wrapped
# like a single case union type
