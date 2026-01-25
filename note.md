16/01
Need to add density estimation to clusters, do not pre-process.
Need to make clusters stricter, somehow this must be measured.

One of the main problems in the tsne space is that separated clusters are not being considered their own, should change with strictness.
Need to write the section on clustering with what is here, change later per new findings and ideas.

On stability
In unsupervised clustering, a lot of the validation is qualitative, so in this case quantitative measures are relative.
Some foundational issues that have been addressed with stability include an estimate of confidence to an item's membership to a cluster, an estimate of confidence to cluster, and an overall estimate of confidence for a clustering of a dataset. Synonymous with cluster stability is its utility in the selection of the optimal number of clusters, which herein we refer to as model selection. 

One clustering method: https://cran.r-project.org/web/packages/fpc/index.html
another: https://cran.r-project.org/web/packages/bootcluster/index.html