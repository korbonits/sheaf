"""Fixed public protein sequences for the model_opt GPU equivalence tests.

Every entry is a public UniProt / PDB sequence (or a trivial synthetic
peptide).  The batch compositions are fixed too: the ESM C kit's ``exact``
guarantee is "same output bytes for the same inputs and batch composition".
"""

from __future__ import annotations

UBIQUITIN = (  # P0CG48 (human ubiquitin, residues 1-76) — the kit README's own
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
LYSOZYME = (  # P00698 mature hen egg-white lysozyme (PDB 1LYZ)
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTP"
    "GSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
)
GFP = (  # P42212 (Aequorea victoria GFP)
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQC"
    "FSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHK"
    "LEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKD"
    "PNEKRDHMVLLEFVTAAGITHGMDELYK"
)
INSULIN_B = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"  # P01308 insulin B chain
TRP_CAGE = "NLYIQWLKDGGPSSGRPPPS"  # PDB 1L2Y (designed miniprotein)
POLY_A = "A" * 12  # synthetic

# One request per entry; the list inside is that request's batch.
BATCHES: dict[str, list[str]] = {
    "b1_ubiquitin": [UBIQUITIN],
    "b1_gfp": [GFP],
    "b1_trpcage": [TRP_CAGE],
    "padded_mixed": [UBIQUITIN, LYSOZYME, INSULIN_B, TRP_CAGE, POLY_A],
    "padded_long": [GFP, LYSOZYME, GFP[::-1]],
    "padded_repeat": [UBIQUITIN] * 8,
}
