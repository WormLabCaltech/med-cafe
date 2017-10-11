library("sleuth")
library("files")
library("optparse")

# optparse is now deprecated, and I should migrate these lines to argparse.
option_list <- list(make_option(c("-d", "--directory"), type='character', default=character(0),
                    help="Please specify the directory")
)
opt = parse_args(OptionParser(option_list=option_list))

try (if(length(opt$d) == 0) stop('Directory cannot be empty'))
try (if(!file.exists(opt$d)) stop('Directory must exist'))

setwd(opt$d)
print(getwd())

#gene info for sleuth
print("Fetching bioMart info:")
mart <- biomaRt::useMart(biomart = "ensembl", dataset = "celegans_gene_ensembl")
t2g <- biomaRt::getBM(attributes = c("ensembl_transcript_id", "ensembl_gene_id",
                                     "external_gene_name"), mart = mart)
print('#renaming genes:')
t2g <- dplyr::rename(t2g, target_id = ensembl_transcript_id,
                     ens_gene = ensembl_gene_id, ext_gene = external_gene_name)


#point to your directory
base_dir <- '../kallisto'

#get ids
sample_id <- dir(file.path(base_dir))
print(sample_id)
kal_dirs <- sapply(sample_id, function(id) file.path(base_dir, id))
print(kal_dirs)
s2c <- read.table("rna_seq_info.txt", header = TRUE, stringsAsFactors= FALSE)
print(head(s2c))

# rename the 'name' column bc sleuth asks for a 'sample' column
s2c <- dplyr::select(s2c, sample=name, genotype, batch, strain)
s2c <- dplyr::mutate(s2c, path=kal_dirs)

so <- sleuth_prep(s2c, ~ strain, target_mapping=t2g)
so <- sleuth_fit(so,~ strain + batch, fit_name='full')

sum(so$fits$full$summary[,2])

# 18667, N2Y1 and N2Y3 look like outliers on PCA plots

outlier1 = (s2c$sample != 'Project_18667_index19') # n2 sample collected by DAA
outlier2 = (s2c$sample != 'Project_N2Y1') # collected by DL
outlier3 = (s2c$sample != 'Project_N2Y3') # collected by DL

no_outliers = s2c[outlier1 & outlier2 & outlier3,]
so <- sleuth_prep(no_outliers, ~ strain, target_mapping=t2g)
so <- sleuth_fit(so,~ strain + batch, fit_name='full')
so <- sleuth_fit(so,~ batch, fit_name='reduced')
so <- sleuth_lrt(so, 'reduced', 'full')
sum(so$fits$full$summary[,2])

#Wald test implementations
for (strain in unique(s2c$strain)){
  if (strain != 'a_n2'){
    beta = paste('strain', strain, sep='')
    so <- sleuth_wt(so, which_beta = beta, which_model='full')
    results_table <- sleuth_results(so, beta, 'full', test_type='wt')
    
    fname = paste(strain, '.csv', sep='')
    write.csv(results_table, paste('sleuth_strains/', fname, sep='/'))
  }
}
  
for (batch in unique(s2c$batch)){
  if (batch != 'a'){
    beta = paste('batch', batch, sep='')
    so <- sleuth_wt(so, which_beta = beta, which_model='full')
    results_table <- sleuth_results(so, beta, 'full', test_type='wt')
    
    fname = paste(batch, '.csv', sep='')
    write.csv(results_table, paste('sleuth_batches/', fname, sep='/'))
  }
}


results_table <- sleuth_results(so, 'reduced:full', test_type='lrt')
fname = paste('lrt', '.csv', sep='')
write.csv(results_table, paste('sleuth_lrt/', fname, sep='/'))

full_model <- extract_model(so, 'full')
fname = paste('model', '.csv', sep='')
write.csv(full_model, paste('sleuth_lrt/', fname, sep='/'))



sleuth_live(so)

