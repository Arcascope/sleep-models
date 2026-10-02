# arcasleep

## Writing for users

Model cards, READMEs and CLI help are read by researchers scoring sleep in
their own studies, often clinical trial data, not by people interested in
time-series methods. Write for that reader: be concise, and say what the
output means for their recordings.

- **Literal over figurative.** Say what happens: "the model's scoring is
  unchanged", not "the model's sleep stands"; "the two differ slightly", not
  "make different trades"; "total sleep time is underestimated by about 11
  minutes", not "comes out short".
- **Exact values for fields.** Name the value a reader checks for: "flagged
  by `is_nap == 1`", not "flagged in `is_nap`".
- **Tight joins.** Join closely related clauses with a semicolon rather than
  a conversational ", and what it does…": "applies a GGIR-style sleep filter;
  the definition depends on recording length".
- **Explicit references.** Prefer "this" or the noun itself to an ambiguous
  "it".
- **Rules get their own line.** A rule that changes the output, such as "≤ 9
  hours: the filter changes nothing", is stated plainly as its own bullet or
  sentence, never folded into a clause of a longer technical sentence.
- **Numbers carry the comparison.** Give the two rates ("82.1% vs 79.8% wake
  specificity") rather than a direction ("leans toward wake"), and make only
  the claims the numbers support: "not significant (p > 0.05)", not
  "indistinguishable".
- **Typography.** Use ≤ and ≥ in prose, not <= and >=; use − for negative
  numbers in tables.
