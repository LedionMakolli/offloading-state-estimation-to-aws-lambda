"""Embed AWS IoT credentials into a C header for the Pico W.

The RP2040 has no filesystem, so mbedTLS cannot open a .pem file at runtime --
the credentials have to be compiled into the binary as string constants. No
format conversion is involved: mbedTLS parses PEM directly
(MBEDTLS_PEM_PARSE_C is on by default), so the bytes are carried through
verbatim and only the packaging changes.

The one detail that trips people up: mbedtls_x509_crt_parse() and
mbedtls_pk_parse_key() require the length to INCLUDE the terminating NUL byte.
Passing strlen() gives a confusing MBEDTLS_ERR_PEM_NO_HEADER_PRESENT. The
generated *_LEN macros use sizeof(), which counts the NUL, so they are correct
by construction.

Security note, worth a paragraph in the thesis: a key compiled into flash can
be read back off the board over SWD, or with `picotool save`. That is
acceptable for a prototype and unacceptable in production, where the key would
live in a secure element (e.g. ATECC608) that performs the private-key
operation without ever revealing it. Documenting this is better than ignoring it.

Usage:
    python firmware/tools/gen_certs.py
    python firmware/tools/gen_certs.py --certs-dir certs --out firmware/src/aws_certs.h
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def find_one(directory: Path, pattern: str, what: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise SystemExit(f"ERROR: no {what} matching '{pattern}' in {directory}")
    if len(matches) > 1:
        raise SystemExit(
            f"ERROR: {len(matches)} files match '{pattern}' in {directory}; "
            f"keep only the set for the Thing you are using:\n  "
            + "\n  ".join(m.name for m in matches)
        )
    return matches[0]


def to_c_string(path: Path, expect_header: str) -> tuple[str, int]:
    """PEM file -> C string literal, one quoted line per source line."""
    text = path.read_text(encoding="utf-8")
    if expect_header not in text:
        raise SystemExit(
            f"ERROR: {path.name} does not contain '{expect_header}'. "
            "Wrong file, or it was re-encoded."
        )
    lines = [ln.rstrip("\r\n") for ln in text.splitlines() if ln.strip()]
    body = "\n".join(f'    "{ln}\\n"' for ln in lines)
    # +1 for the NUL that the C compiler appends and that mbedTLS requires.
    n_bytes = sum(len(ln) + 1 for ln in lines) + 1
    return body, n_bytes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--certs-dir", type=Path, default=REPO / "certs")
    ap.add_argument("--out", type=Path, default=REPO / "firmware" / "src" / "aws_certs.h")
    ap.add_argument("--root-ca", default="AmazonRootCA1.pem",
                    help="AmazonRootCA1 for RSA device certs, RootCA3 for ECC")
    args = ap.parse_args()

    d = args.certs_dir
    if not d.is_dir():
        raise SystemExit(f"ERROR: {d} does not exist")

    root_ca = d / args.root_ca
    if not root_ca.exists():
        raise SystemExit(f"ERROR: {root_ca} not found")
    cert = find_one(d, "*-certificate.pem.crt", "device certificate")
    key = find_one(d, "*-private.pem.key", "private key")

    ca_body, ca_len = to_c_string(root_ca, "BEGIN CERTIFICATE")
    crt_body, crt_len = to_c_string(cert, "BEGIN CERTIFICATE")
    key_body, key_len = to_c_string(key, "PRIVATE KEY")

    header = f'''/* GENERATED FILE - DO NOT EDIT, DO NOT SHARE.
 *
 * Produced by firmware/tools/gen_certs.py from {d.name}/.
 * Contains the device PRIVATE KEY. It is excluded by .gitignore; keep it that
 * way, and do not paste this file into a report, an issue, or a chat window.
 *
 * Lengths include the terminating NUL, which mbedTLS requires. Pass the
 * *_LEN macros directly to mbedtls_x509_crt_parse / mbedtls_pk_parse_key.
 */
#ifndef AWS_CERTS_H
#define AWS_CERTS_H

/* Amazon root CA ({root_ca.name}) - validates the AWS IoT server. */
static const char AWS_ROOT_CA[] =
{ca_body};
#define AWS_ROOT_CA_LEN (sizeof(AWS_ROOT_CA))

/* Device certificate - this board's identity, presented to AWS. */
static const char AWS_DEVICE_CERT[] =
{crt_body};
#define AWS_DEVICE_CERT_LEN (sizeof(AWS_DEVICE_CERT))

/* Device private key - SECRET. Never leaves the device in normal operation. */
static const char AWS_PRIVATE_KEY[] =
{key_body};
#define AWS_PRIVATE_KEY_LEN (sizeof(AWS_PRIVATE_KEY))

#endif /* AWS_CERTS_H */
'''

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header, encoding="utf-8")

    total = ca_len + crt_len + key_len
    print(f"wrote {args.out.relative_to(REPO)}")
    print(f"  root CA        {root_ca.name:<52} {ca_len:5d} B")
    print(f"  device cert    {cert.name[:52]:<52} {crt_len:5d} B")
    print(f"  private key    {key.name[:52]:<52} {key_len:5d} B")
    print(f"  total in flash {'':<52} {total:5d} B")
    print("\n  no format conversion performed - mbedTLS parses PEM directly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
