#!/usr/bin/env bash
#
# run.sh -- the old name of capture.sh, kept so that notes and habits written
# for it still work. Delete after the next release.
#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, The valubench authors. See LICENSE.

echo "run.sh: renamed to tools/capture.sh; running that" >&2
exec "$(dirname "$0")/capture.sh" "$@"
