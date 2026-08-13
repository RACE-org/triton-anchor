module {
  func.func @nested(%condition: i1) {
    scf.if %condition {
      "smt.deep"() : () -> ()
    }
    func.return
  }
}
