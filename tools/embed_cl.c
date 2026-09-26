/*
 * embed_cl.c -- turn a device kernel source file into a C byte array.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026, The valubench authors. See LICENSE.
 *
 * A device kernel has to reach the runtime compiler as a string, but writing
 * it as a string literal by hand costs syntax highlighting, escaping, and any
 * hope of a readable diff. So each piece -- a dialect header, the primitives,
 * an algorithm core -- lives in a real file under src/kernels/gpu and this
 * emits the header that embeds it. The backends concatenate the pieces into a
 * program at run time.
 *
 * Given a file name, the output starts with a #line directive naming it. The
 * pieces are compiled as one concatenated program, so without it a compiler
 * error would cite a line of the whole program rather than of the file a
 * person would open.
 *
 * The result is generated into $(BUILD) on every build and is not committed:
 * it is a pure function of its input, and this program needs nothing but the C
 * compiler the build already requires, so a second copy in the tree could only
 * ever be a thing to keep in sync. The binary stays self-contained either way
 * -- the array is compiled in, so there is no data file to install and nothing
 * to locate at run time.
 *
 * Written in C rather than as a shell or Python step so the build depends on
 * nothing beyond the compiler it already needs. xxd would have done the job but
 * lives in vim-common, which is exactly the kind of incidental dependency
 * the dependency budget rules out.
 *
 * Emitted as a byte array rather than a string literal on purpose. C99 only
 * guarantees string literals up to 4095 characters (5.2.4.1); real compilers
 * accept far more, but -Wpedantic rightly complains, and a kernel is easily
 * longer than that once it carries comments. An array initialiser has no such
 * limit and stays warning-clean everywhere.
 *
 * The consequence is that the generated header is machine output and not worth
 * reading. Review changes in the source file, which is the source of truth.
 *
 *   Usage: embed_cl SYMBOL GUARD [FILENAME] < input > output.h
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t count;

static void emit(int c)
{
    if (count % 16 == 0)
        fputs("    ", stdout);
    printf("0x%02x,", (unsigned) (unsigned char) c);
    count++;
    if (count % 16 == 0)
        fputc('\n', stdout);
}

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: %s SYMBOL GUARD [FILENAME] < input > output.h\n",
                argv[0]);
        return 2;
    }

    const char *symbol = argv[1];
    const char *guard = argv[2];
    const char *filename = argc == 4 ? argv[3] : NULL;

    printf("/*\n");
    printf(" * %s -- GENERATED FILE, DO NOT EDIT.\n", guard);
    printf(" *\n");
    printf(" * Produced by tools/embed_cl.c from the kernel source, into the\n");
    printf(" * build directory. Not committed and not edited: change the\n");
    printf(" * source file and rebuild.\n");
    printf(" *\n");
    printf(" * Machine output -- review changes in the source, not here. A\n");
    printf(" * byte array rather than a string literal because C99 only\n");
    printf(" * guarantees 4095-character string literals and a kernel exceeds\n");
    printf(" * that; an array initialiser has no such limit.\n");
    printf(" *\n");
    /* No licence notice here on purpose. This header is generated into the
       build directory, is gitignored, and is never redistributed as source;
       the source it comes from carries the SPDX tag, and a binary is covered
       by shipping LICENSE. A notice here would only be one more place to
       forget when the licence changes -- as it just was. */
    printf(" * Licence follows the source this was generated from.\n");
    printf(" */\n\n");
    printf("#ifndef %s\n", guard);
    printf("#define %s\n\n", guard);
    printf("static const char %s[] = {\n", symbol);

    if (filename) {
        char line[512];
        snprintf(line, sizeof line, "#line 1 \"%s\"\n", filename);
        for (const char *p = line; *p; p++)
            emit(*p);
    }

    int c;
    while ((c = fgetc(stdin)) != EOF)
        emit(c);

    /* NUL terminate: callers pass this straight to a runtime compiler as a C
       string. */
    if (count % 16 == 0)
        fputs("    ", stdout);
    printf("0x00\n");

    printf("};\n\n");
    printf("#endif /* %s */\n", guard);
    return 0;
}
