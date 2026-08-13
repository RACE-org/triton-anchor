module {
  func.func @illegal_type(%value: !smt.bad) {
    func.return
  }
}
