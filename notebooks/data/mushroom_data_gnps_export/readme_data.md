Mushroom dataset from GNPS with file names renamed for brevity:

**Files used:**

- quantification_table.csv; equally named in gnps export.
- spectra.mgf ; this is the specs_ms.mgf file renamed. Lot's of empty msms spectra in there to be cleaned out first.
- metadata.tsv ; this is the "FEATURE-BASED-MOLECULAR-NETWORKING-2b86dd35-view_metadata-main.tsv" file renamed

**Data Relationships:**

- metadata and quantification table are linked via sample names, which are columns (with peak area suffix) in the quantification table and entries in the metadata table
- spectra and quantificationt table are linked via scan numbers, which are entry identifiers in spectra and rowids in quantification table

**Processing Requirement**

Spectral filtering is required since a lot of the entries contain insufficient or no informaiton. Hence iloc alignment should not be relied upon in principle until after processing.