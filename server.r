library(shiny)
library(chorddiag)

new_col_names <- NULL
for (i in 1:122){
  if (i<=18) {new_col_names[i] <- paste("DG ",i)}
  else if (i>18 & i<=43) {new_col_names[i] <- paste("CA3 ",i)}
  else if (i>43 & i<=48) {new_col_names[i] <- paste("CA2 ",i)}
  else if (i>48 & i<=88) {new_col_names[i] <- paste("CA1 ",i)}
  else if (i>88 & i<=91) {new_col_names[i] <- paste("SUB ",i)}
  else {new_col_names[i] <- paste("EC ",i)}
}
smallworld_col_names <- c(1:122)

inhib_mat <- read.csv(file="conn_mat0.csv", header = FALSE, row.names=NULL)
colnames(inhib_mat) <- new_col_names
row.names(inhib_mat) = c(colnames(inhib_mat))

excit_mat <- read.csv(file="conn_mat1.csv", header = FALSE, row.names=NULL)
colnames(excit_mat) <- new_col_names
row.names(excit_mat) = c(colnames(excit_mat))

#spike_sync_mat <- read.csv(file="spike_sync_matrix.csv", header = FALSE, row.names=NULL)
#colnames(spike_sync_mat) <- longer_col_names
#row.names(spike_sync_mat) = c(colnames(spike_sync_mat))

small_world_mat <- read.csv(file="small_world.csv", header = FALSE, row.names=NULL)
colnames(small_world_mat) <- smallworld_col_names
row.names(small_world_mat) = c(colnames(smallworld_col_names))

#spike_dist_mat <- read.csv(file="spike_distance_mat.csv", header = FALSE, row.names=NULL)
#colnames(spike_dist_mat) <- longer_col_names
#row.names(spike_dist_mat) = c(colnames(spike_dist_mat))
#inverse_spike_dist_mat <- 1/spike_dist_mat
#inverse_spike_dist_mat[inverse_spike_dist_mat == Inf] <- 0

cell_names <- read.csv("cell_names.csv", header=FALSE, row.names=NULL, stringsAsFactors = FALSE)


shinyServer(function(input, output) {
  

  output$distPlot <- renderChorddiag({
    adj_names = cell_names[input$range[1]:input$range[2]]
    cell_names_plot = as.character(cell_names[2,input$range[1]:input$range[2]])
    if (input$select_matrix == "Excitatory") {
      chorddiag(as.matrix(excit_mat[input$range[1]:input$range[2],input$range[1]:input$range[2]]), showTicks = F, groupnameFontsize = 12, groupnamePadding = 10, margin = 90, tooltipNames = cell_names_plot)
    } else if (input$select_matrix == "Inhibitory"){
      chorddiag(as.matrix(inhib_mat[input$range[1]:input$range[2],input$range[1]:input$range[2]]), showTicks = F, groupnameFontsize = 10, groupnamePadding = 10, margin = 90, tooltipNames = cell_names_plot)
    } else if (input$select_matrix == "Spike Sync") {
      chorddiag(as.matrix(spike_sync_mat[input$range[1]:input$range[2],input$range[1]:input$range[2]]), showTicks = F, groupnameFontsize = 14, groupnamePadding = 10, margin = 90, tooltipNames = adj_names)
    }
      else if (input$select_matrix == "Small World") {
      chorddiag(as.matrix(small_world_mat[input$range[1]:input$range[2],input$range[1]:input$range[2]]), showTicks = F, groupnameFontsize = 14, groupnamePadding = 10, margin = 90, tooltipNames = smallworld_col_names[input$range[1]:input$range[2]])
    } else {
      chorddiag(as.matrix(inverse_spike_dist_mat[input$range[1]:input$range[2],input$range[1]:input$range[2]]), showTicks = F, groupnameFontsize = 14, groupnamePadding = 10, margin = 90, tooltipNames = adj_names)
    }
    
  })

  
})
