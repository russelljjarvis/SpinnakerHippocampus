library(shiny)
library(chorddiag)

new_col_names <- c(1:122)
inhib_mat <- read.csv(file="conn_mat1.csv", header = FALSE)
colnames(inhib_mat) <- new_col_names
row.names(inhib_mat) = c(colnames(inhib_mat))

excit_mat <- read.csv(file="conn_mat0.csv", header = FALSE)
colnames(excit_mat) <- new_col_names
row.names(excit_mat) = c(colnames(excit_mat))

names <- NULL
for (i in 1:122){
  names[i] <- ((paste("Neuron ",i)))
}


shinyServer(function(input, output) {
  

  output$distPlot <- renderChorddiag({
    adj_names=names[input$range[1]:input$range[2]]
    if (input$select_matrix == "Excitatory"){
      chorddiag(as.matrix(excit_mat[input$range[1]:input$range[2],input$range[1]:input$range[2]]), showTicks = F, groupnameFontsize = 14, groupnamePadding = 10, margin = 90, tooltipNames = adj_names)
    }else{
      chorddiag(as.matrix(inhib_mat[input$range[1]:input$range[2],input$range[1]:input$range[2]]), showTicks = F, groupnameFontsize = 14, groupnamePadding = 10, margin = 90, tooltipNames = adj_names)
    }
    
  })

  
})
