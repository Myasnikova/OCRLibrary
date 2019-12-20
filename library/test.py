import coverage as cov
import FilteredImage as fi
import ContouredImage as ci

coverage = cov.Coverage()
coverage.erase()
coverage.start()

#fi.test()
ci.test()

coverage.stop()
coverage.save()
coverage.html_report()