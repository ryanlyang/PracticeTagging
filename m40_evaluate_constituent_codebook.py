#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from m40_constituent_codebook import build_eval_parser, cli_eval


if __name__ == "__main__":
    parser = build_eval_parser()
    args = parser.parse_args()
    raise SystemExit(cli_eval(args))
