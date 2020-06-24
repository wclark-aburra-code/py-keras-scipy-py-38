from fft_convolution import Filter, unit_impulse, input_side_convolve, MonoFilter, convolve
# EffectChain
import numpy    # type: ignore
import unittest # type: ignore


multitap_delay = numpy.array([1., 0., 0.6, 0., 0.4, 0., 0.2])
slapback_delay = numpy.array([1., 0.8])
basic_reverb = numpy.array([0., .1, -.1, .2, -.1, .2, -.1])
turn_up = numpy.array([2.])
lowpass_filter = numpy.array([1., 0.2, 0.2, 0.2, 0.2, 0.2]) # moving average
linear_phase_highpass_filter = numpy.array([-0.2, -0.2, 1., -0.2, -0.2])
mute = numpy.array([0.])
differential = numpy.array([1., -1.])
hpf_differential = numpy.array([1., -.1]) # like a highpass filter


test_signal = numpy.array([1.,0.,-1.,0])

class FilterTest(unittest.TestCase):
    def testConvolveTurnUp(self):
        f0 = MonoFilter(turn_up)
        i0 = unit_impulse()
        total_length = len(i0) + len(turn_up) - 1 
        wet_signal = convolve(f0, i0)
        clipped_signal = wet_signal[0:total_length]
        self.assertTrue(numpy.allclose(clipped_signal, numpy.array([2.])))
    # FAILS
    #def testConvolveDelay(self):
    #    f0 = MonoFilter(slapback_delay)
    #    i0 = [1., 0., -1., -.1]
    #    total_length = len(slapback_delay) + len(i0) - 1 
    #    wet_signal = convolve(f0, i0)
    #    clipped_signal = wet_signal[0:total_length]
    #    self.assertTrue(numpy.allclose(clipped_signal, numpy.array([ 1.  ,  0.8 , -1.  , -0.9 , -0.08])))
    def testStepResponse(self):
        bigLowpassIR = numpy.array([1.,1.,1.,1.])
        filterLP = MonoFilter(bigLowpassIR)
        self.assertTrue(numpy.allclose(filterLP.step_response(), numpy.array([1., 2., 3., 4., 3., 2., 1.])))
        self.assertEqual(round(filterLP.normalization_factor()), 4.0)

    # FAILS
    #def testInputSideConvolution(self):
    #    input_side_convolved_sig = input_side_convolve(test_signal, linear_phase_highpass_filter)
    #    fourier_convolved_sig = convolve(MonoFilter(linear_phase_highpass_filter), test_signal)
    #    self.assertTrue(numpy.allclose(fourier_convolved_sig[0:8], input_side_convolved_sig))
     

if __name__ == '__main__':
    unittest.main()