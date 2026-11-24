# Copyright 2026 Scott Friedman
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

"""
A stand-in for an expensive evaluator and a fake model, so the search-quality
capabilities (#57) can be built and A/B-tested with no AWS account (#46).

- ``problem``: the evaluator. ``evaluate`` / ``evaluate_stage1`` / ``evaluate_stage2``
  / ``evaluate_heldout``, a known optimum, seeded noise, injectable faults and
  planted defects in ``defects/``.
- ``fake_llm``: ``provider: oeb-fake``, registered when openevolve_bedrock is imported.
- ``evaluator.py``: the file to pass to ``oe-bedrock run``; ``config.yaml``: a run config.
"""

from openevolve_bedrock.standin import problem  # noqa: F401
