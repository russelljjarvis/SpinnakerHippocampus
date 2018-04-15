library(shiny)
library(chorddiag)

shinyUI(fluidPage(
  br(),
  br(),
  radioButtons('select_matrix',"Select Matrix",inline = TRUE,
               choices = c("Excitatory","Inhibitory","Small World"),
               selected = 'Small World'),
  sliderInput('range', "Range of neurons:",
              min = 1, max = 122,
              value = c(1,85), ticks=FALSE),
  chorddiagOutput("distPlot", height = 600)
))