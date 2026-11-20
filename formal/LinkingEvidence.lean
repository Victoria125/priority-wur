/-!
Formal toy model of the *requirements* for a strain-resolved linking benchmark.

This formalizes safety invariants (absence of critical evidence => NOT ready).
It does NOT certify Python equivalence, biological evidence, data quality,
or model performance. The Python code and its tests provide a separate layer.
-/

structure LinkingEvidence where
  spectrumToStrain : Bool
  strainToBgc : Bool
  independentlyVerifiedPairs : Bool
  strainVariation : Bool
deriving Repr

def readyForBenchmark (e : LinkingEvidence) : Bool :=
  e.spectrumToStrain && e.strainToBgc &&
  e.independentlyVerifiedPairs && e.strainVariation

theorem noSpectrumMappingCannotBeReady (e : LinkingEvidence)
    (h : e.spectrumToStrain = false) : readyForBenchmark e = false := by
  simp [readyForBenchmark, h]

theorem noBGCMappingCannotBeReady (e : LinkingEvidence)
    (h : e.strainToBgc = false) : readyForBenchmark e = false := by
  simp [readyForBenchmark, h]

theorem noVerifiedPairsCannotBeReady (e : LinkingEvidence)
    (h : e.independentlyVerifiedPairs = false) :
    readyForBenchmark e = false := by
  simp [readyForBenchmark, h]

theorem noStrainVariationCannotBeReady (e : LinkingEvidence)
    (h : e.strainVariation = false) : readyForBenchmark e = false := by
  simp [readyForBenchmark, h]

-- The count of PoDP literature links is irrelevant if there is no
-- independently curated, spectrum-level, strain-resolved mapping.
theorem anyNumberOfPublishedLinksIsInsufficient (reportedLinks : Nat) :
    readyForBenchmark {
      spectrumToStrain := false
      strainToBgc := false
      independentlyVerifiedPairs := false
      strainVariation := false
    } = false := by
  simp [readyForBenchmark]
