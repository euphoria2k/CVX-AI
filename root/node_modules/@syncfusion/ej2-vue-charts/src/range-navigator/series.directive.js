var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
        return extendStatics(d, b);
    }
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
import Vue from 'vue';
import { EJComponentDecorator } from '@syncfusion/ej2-vue-base';
var RangenavigatorSeriesCollectionDirective = /** @class */ (function (_super) {
    __extends(RangenavigatorSeriesCollectionDirective, _super);
    function RangenavigatorSeriesCollectionDirective() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    RangenavigatorSeriesCollectionDirective.prototype.render = function () {
        return;
    };
    RangenavigatorSeriesCollectionDirective = __decorate([
        EJComponentDecorator({})
    ], RangenavigatorSeriesCollectionDirective);
    return RangenavigatorSeriesCollectionDirective;
}(Vue));
export { RangenavigatorSeriesCollectionDirective };
export var RangenavigatorSeriesCollectionPlugin = {
    name: 'e-rangenavigator-series-collection',
    install: function (Vue) {
        Vue.component(RangenavigatorSeriesCollectionPlugin.name, RangenavigatorSeriesCollectionDirective);
    }
};
var RangenavigatorSeriesDirective = /** @class */ (function (_super) {
    __extends(RangenavigatorSeriesDirective, _super);
    function RangenavigatorSeriesDirective() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    RangenavigatorSeriesDirective.prototype.render = function () {
        return;
    };
    RangenavigatorSeriesDirective = __decorate([
        EJComponentDecorator({})
    ], RangenavigatorSeriesDirective);
    return RangenavigatorSeriesDirective;
}(Vue));
export { RangenavigatorSeriesDirective };
export var RangenavigatorSeriesPlugin = {
    name: 'e-rangenavigator-series',
    install: function (Vue) {
        Vue.component(RangenavigatorSeriesPlugin.name, RangenavigatorSeriesDirective);
    }
};
