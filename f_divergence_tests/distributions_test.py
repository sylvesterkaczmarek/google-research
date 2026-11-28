# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Expo-1D rejection-sampling envelope."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from f_divergence_tests import distributions


class Expo1dSamplesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('narrow_peak', 4.0, 0.01, 100.0),
      ('boundary_peak', 0.0, 0.01, 100.0),
      ('broad_peak', 4.0, 1.6, 90.0),
      ('null', 4.0, 0.01, 0.0),
  )
  def test_rejection_envelope_bounds_density(self, location, scale, multiplier):
    with mock.patch.object(distributions, 'rejection_sampling') as sampler:
      distributions.Expo1dSamples(
          location=location,
          scale=scale,
          multiplier=multiplier,
          n_samples=1,
          key_seed=42,
      )
    self.assertLen(sampler.call_args_list, 2)
    for call in sampler.call_args_list:
      args = call.kwargs
      points = np.append(
          np.linspace(args['minval'], args['maxval'], 10001), location
      )
      density = args['density_fn'](points)
      self.assertLessEqual(float(np.max(density)), args['density_max'])


if __name__ == '__main__':
  absltest.main()
