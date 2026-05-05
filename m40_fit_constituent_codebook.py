#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from m40_constituent_codebook import build_fit_parser, cli_fit


if __name__ == "__main__":
    parser = build_fit_parser()
    args = parser.parse_args()
    raise SystemExit(cli_fit(args))
