library("sleuth")
library("files")
library("optparse")

#gene info for sleuth
print("Fetching bioMart info:")
mart <- biomaRt::useMart(host = "metazoa.ensembl.org", 
                         biomart = "metazoa_mart",
                         dataset = "celegans_eg_gene")
t2g <- biomaRt::getBM(attributes =  c("ensembl_transcript_id",
                                      "ensembl_gene_id", "external_gene_name",
                                      "description", "transcript_biotype"),
                      mart = mart)
print('#renaming genes:')
t2g <- dplyr::rename(t2g, target_id = ensembl_transcript_id,
                     ens_gene = ensembl_gene_id, ext_gene = external_gene_name)


#point to your directory
print('Loading projects')
base_dir <- '../kallisto'

#get ids
sample_id <- dir(file.path(base_dir))
kal_dirs <- sapply(sample_id, function(id) file.path(base_dir, id))
s2c <- read.table("rna_seq_info.txt", header = TRUE, stringsAsFactors= FALSE)

# rename the 'name' column bc sleuth asks for a 'sample' column
s2c <- dplyr::select(s2c, sample=name, genotype, project, sync_method, person_collected_worms, library_kind, strain,
                     allele1, allele2, date)
s2c <- dplyr::mutate(s2c, path=kal_dirs)
s2c = s2c[(s2c$project != 'female_state'),]
# s2c = s2c[(s2c$project != 'single_worm_benchmarking'),]

print('Fitting model')
so <- sleuth_prep(s2c, ~ strain, target_mapping=t2g)
so <- sleuth_fit(so,~ strain + person_collected_worms + library_kind, fit_name='full')

#Wald test implementations
print("Differential Expression Testing")
for (strain in unique(s2c$strain)){
  if (strain != 'a_n2'){
    beta = paste('strain', strain, sep='')
    so <- sleuth_wt(so, which_beta = beta, which_model='full')
    results_table <- sleuth_results(so, beta, 'full', test_type='wt')
    
    fname = paste(strain, '.csv', sep='')
    write.csv(results_table, paste('sleuth_strains', fname, sep='/'))
  }
}



# go live on the first model built
sleuth_live(so)
