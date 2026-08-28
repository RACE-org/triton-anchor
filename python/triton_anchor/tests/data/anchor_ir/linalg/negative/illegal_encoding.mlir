module {
  func.func @illegal_encoding(
      %value: tensor<4xf32, #smt.encoding>) {
    func.return
  }
}
