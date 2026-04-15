#!/usr/bin/env python3
"""Entry point for m23 multihyp experiment.

This file intentionally imports the implementation from the v2 module, which
contains the m23 multihyp/selector extensions.
"""

from offline_reconstructor_joint_dualview_seq2seq_nexttoken_v2 import main


if __name__ == "__main__":
    main()
