library(readr)

output_static <- read_csv("~/PycharmProjects/SwarmIntelligence/results/output-files/heart/output_mlp_fips_static_heart_21.csv",
                          col_names = FALSE)
output_distance <- read_csv("~/PycharmProjects/SwarmIntelligence/results/output-files/heart/output_mlp_fips_distance_heart_5.csv",
                            col_names = FALSE)
output_fitness <- read_csv("~/PycharmProjects/SwarmIntelligence/results/output-files/heart/output_mlp_fips_fitness_heart_14.csv",
                           col_names = FALSE)
output_classic <- read_csv("~/PycharmProjects/SwarmIntelligence/results/output-files/heart/output_mlp_pso_heart_24.csv",
                           col_names = FALSE)
output_cs <- read_csv("~/PycharmProjects/SwarmIntelligence/results/output-files/heart/output_mlp_cs_heart_26.csv",
                      col_names = FALSE)
output_ring <- read_csv("~/PycharmProjects/SwarmIntelligence/results/output-files/heart/output_mlp_ring_heart_21.csv",
                        col_names = FALSE)
output_pso_w <- read_csv("~/PycharmProjects/SwarmIntelligence/results/output-files/heart/output_mlp_psow_heart_23.csv",
                         col_names = FALSE)
output_backpropagation <- 
  read_csv("~/PycharmProjects/SwarmIntelligence/results/output-files/heart/output_backpropagation_heart_4.csv",
           col_names = FALSE)

names(output_static) <- c('iteration','gbest_error','n_hiddens','n_connections')
names(output_distance) <- c('iteration','gbest_error','n_hiddens','n_connections')
names(output_fitness) <- c('iteration','gbest_error','n_hiddens','n_connections')
names(output_classic) <- c('iteration','gbest_error','n_hiddens','n_connections')
names(output_cs) <- c('iteration','gbest_error','n_hiddens','n_connections')
names(output_ring) <- c('iteration','gbest_error','n_hiddens','n_connections')
names(output_pso_w) <- c('iteration','gbest_error','n_hiddens','n_connections')
names(output_backpropagation) <- c('iteration','gbest_error','n_hiddens','n_connections')

par(xpd = T, mar = par()$mar + c(0,0,0,6))
plot(output_backpropagation$iteration[0:600], output_backpropagation$gbest_error[0:600], 
     type='l', lty=1, col = "purple", ylim=c(0, 0.55), 
     xlab='iteration', ylab='error', main='Heart Dataset')
lines(output_cs$iteration, output_cs$gbest_error, type='l', lty=1, col = "darkred")
lines(output_classic$iteration, output_classic$gbest_error, type='l', lty=1, col = "green")
lines(output_ring$iteration, output_ring$gbest_error, type='l', lty=1, col = "darkblue")
lines(output_pso_w$iteration, output_pso_w$gbest_error, type='l', lty=2, col = "darkorchid4")
lines(output_static$iteration, output_static$gbest_error, type='l', lty=2, col = "gold4")
lines(output_fitness$iteration, output_fitness$gbest_error, type='l', lty=2, col = "blue")
lines(output_distance$iteration, output_distance$gbest_error, type='l', lty=2, col = "red")

legend(650,0.4,
       legend=c("backprop", "cuckoo", "classic", "ring","pso_w", "FIPS", "wFIPS", "wdFIPS"), 
       col=c("purple","darkred","green","darkblue","darkorchid4","gold4","blue","red"), 
       lty=c(1,1,1,1,2,2,2,2), ncol=1)
par(mar=c(5, 4, 4, 2) + 0.1)
