#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from m40_constituent_codebook import build_sweep_parser, cli_sweep


if __name__ == "__main__":
    parser = build_sweep_parser()
    args = parser.parse_args()
    raise SystemExit(cli_sweep(args))
