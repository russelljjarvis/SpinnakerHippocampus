library(shiny)
library(chorddiag)

shinyUI(fluidPage(
  br(),
  br(),
  radioButtons('select_matrix',"Select Matrix",inline = TRUE,
               choices = c("Excitatory","Inhibitory"),
               selected = 'Excitatory'),
  sliderInput('range', "Range:",
              min = 1, max = 122,
              value = c(1,22)),
  chorddiagOutput("distPlot", height = 600)
))