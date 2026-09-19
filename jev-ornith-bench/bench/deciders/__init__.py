"""The arms under test.

Every arm answers the same `Question` about the same `state`. They differ only
in the machinery that produces the answer:

  solo         Ornith is asked in prose and the reply is parsed. The realistic
               baseline, and the only arm that can emit something outside the
               type.
  constrained  Ornith again, but decoding is constrained to a JSON schema.
               Also cannot violate the type. This is the control that
               separates "Jev is good at deciding" from "any schema helps".
  jev          TypeSafe's System One model answers the typed question directly.

Without `constrained` the experiment cannot distinguish the two explanations,
and the headline result would be uninterpretable. It is not an optional extra.
"""

from bench.deciders.base import Decider
from bench.deciders.constrained import ConstrainedDecider
from bench.deciders.jev import JevDecider
from bench.deciders.solo import SoloDecider

ARMS = {
    "solo": SoloDecider,
    "constrained": ConstrainedDecider,
    "jev": JevDecider,
}

__all__ = ["Decider", "SoloDecider", "ConstrainedDecider", "JevDecider", "ARMS"]
