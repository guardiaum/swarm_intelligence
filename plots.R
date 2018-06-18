library(readr)

output_mlp_pso <- read_csv("PycharmProjects/Backpropagation/output_mlp_pso_cancer.csv", 
                           col_names = FALSE)
output_mlp_pso_w <- read_csv("PycharmProjects/Backpropagation/output_mlp_pso_w_cancer.csv", 
                           col_names = FALSE)

names(output_mlp_pso) <- c('iteration','gbest_error','n_hiddens','n_connections')
names(output_mlp_pso_w) <- c('iteration','gbest_error','n_hiddens','n_connections')

plot(output_mlp_pso$iteration, output_mlp_pso$gbest_error, type='l', 
     col = "red", ylim=c(0, 0.7), xlab='iteration', ylab='error', main='Cancer Dataset')
lines(output_mlp_pso_w$iteration, output_mlp_pso_w$gbest_error, type='l', col = "blue")
legend(400,0.7,legend=c("MLP_PSO","MLP_PSO_W"), col=c("red","blue"),lty=c(1,1), ncol=1)
